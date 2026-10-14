// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Compare small composite integer keys directly inside hash buckets.

use super::{GroupValues, HashValue};
use ahash::RandomState;
use arrow::array::{ArrayRef, AsArray, PrimitiveArray};
use arrow::datatypes::{
    DataType, Int8Type, Int16Type, Int32Type, Int64Type, SchemaRef, UInt8Type,
    UInt16Type, UInt32Type, UInt64Type,
};
use datafusion_common::Result;
use datafusion_expr::EmitTo;
use hashbrown::hash_table::{Entry, HashTable};
use std::{mem, sync::Arc};

#[cfg(test)]
#[path = "packed_test.rs"]
mod tests;

pub(super) struct PackedGroups {
    schema: SchemaRef,
    layout: Vec<(u32, u32)>,
    bits: u32,
    mask: u128,
    limit: usize,
    map: HashTable<u128>,
    wide: Option<HashTable<(u128, usize)>>,
    keys: Vec<u128>,
    scratch: Vec<u128>,
    random: RandomState,
}

impl PackedGroups {
    pub(super) fn try_new(schema: SchemaRef) -> Option<Self> {
        let mut bits = 0;
        let mut layout = Vec::new();
        for field in &schema.fields {
            let width = match field.data_type() {
                DataType::Int8 | DataType::UInt8 => 8,
                DataType::Int16 | DataType::UInt16 => 16,
                DataType::Int32 | DataType::UInt32 => 32,
                DataType::Int64 | DataType::UInt64 => 64,
                _ => return None,
            };
            layout.push((bits, width));
            bits += width + 1;
        }
        if !(2..=8).contains(&layout.len()) || bits > 100 {
            return None;
        }
        Some(Self {
            schema,
            layout,
            bits,
            mask: (1u128 << bits) - 1,
            limit: 1usize.checked_shl(128 - bits).unwrap_or(usize::MAX),
            map: HashTable::new(),
            wide: None,
            keys: Vec::new(),
            scratch: Vec::new(),
            random: crate::aggregates::AGGREGATION_HASH_SEED,
        })
    }

    fn promote(&mut self) {
        let mut wide = HashTable::with_capacity(self.map.len());
        let (mask, bits, random) = (self.mask, self.bits, &self.random);
        for entry in mem::take(&mut self.map) {
            let key = entry & mask;
            wide.insert_unique(
                (key as i128).hash(random),
                (key, (entry >> bits) as usize),
                |&(key, _)| (key as i128).hash(random),
            );
        }
        self.wide = Some(wide);
    }

    fn arrays(&self, keys: &[u128]) -> Vec<ArrayRef> {
        self.layout
            .iter()
            .zip(&self.schema.fields)
            .map(|(&(shift, width), field)| {
                macro_rules! build {
                    ($t:ty, $native:ty) => {
                        Arc::new(PrimitiveArray::<$t>::from_iter(keys.iter().map(
                            |key| {
                                (key & (1u128 << (shift + width)) == 0)
                                    .then_some((key >> shift) as $native)
                            },
                        ))) as ArrayRef
                    };
                }
                match field.data_type() {
                    DataType::Int8 => build!(Int8Type, i8),
                    DataType::Int16 => build!(Int16Type, i16),
                    DataType::Int32 => build!(Int32Type, i32),
                    DataType::Int64 => build!(Int64Type, i64),
                    DataType::UInt8 => build!(UInt8Type, u8),
                    DataType::UInt16 => build!(UInt16Type, u16),
                    DataType::UInt32 => build!(UInt32Type, u32),
                    DataType::UInt64 => build!(UInt64Type, u64),
                    _ => unreachable!(),
                }
            })
            .collect()
    }
}

impl GroupValues for PackedGroups {
    fn intern(&mut self, cols: &[ArrayRef], groups: &mut Vec<usize>) -> Result<()> {
        self.scratch.clear();
        self.scratch.resize(cols[0].len(), 0);
        for (col, &(shift, width)) in cols.iter().zip(&self.layout) {
            let mask = (1u128 << width) - 1;
            macro_rules! encode {
                ($t:ty) => {
                    for (key, value) in
                        self.scratch.iter_mut().zip(col.as_primitive::<$t>().iter())
                    {
                        *key |= match value {
                            Some(value) => (value as u128 & mask) << shift,
                            None => 1u128 << (shift + width),
                        };
                    }
                };
            }
            match col.data_type() {
                DataType::Int8 => encode!(Int8Type),
                DataType::Int16 => encode!(Int16Type),
                DataType::Int32 => encode!(Int32Type),
                DataType::Int64 => encode!(Int64Type),
                DataType::UInt8 => encode!(UInt8Type),
                DataType::UInt16 => encode!(UInt16Type),
                DataType::UInt32 => encode!(UInt32Type),
                DataType::UInt64 => encode!(UInt64Type),
                _ => unreachable!(),
            }
        }
        groups.clear();
        for row in 0..self.scratch.len() {
            if self.wide.is_none() && self.keys.len() == self.limit {
                self.promote();
            }
            let key = self.scratch[row];
            let random = &self.random;
            let hash = (key as i128).hash(random);
            let next = self.keys.len();
            let group = if let Some(wide) = &mut self.wide {
                match wide.entry(
                    hash,
                    |&(value, _)| value == key,
                    |&(value, _)| (value as i128).hash(random),
                ) {
                    Entry::Occupied(entry) => entry.get().1,
                    Entry::Vacant(entry) => {
                        entry.insert((key, next));
                        next
                    }
                }
            } else {
                let (mask, bits) = (self.mask, self.bits);
                match self.map.entry(
                    hash,
                    |value| value & mask == key,
                    |value| ((value & mask) as i128).hash(random),
                ) {
                    Entry::Occupied(entry) => (entry.get() >> bits) as usize,
                    Entry::Vacant(entry) => {
                        entry.insert(key | ((next as u128) << bits));
                        next
                    }
                }
            };
            if group == next {
                self.keys.push(key);
            }
            groups.push(group);
        }
        Ok(())
    }

    fn size(&self) -> usize {
        size_of::<Self>()
            + self.map.allocation_size()
            + self.wide.as_ref().map_or(0, HashTable::allocation_size)
            + (self.keys.capacity() + self.scratch.capacity()) * size_of::<u128>()
            + self.layout.capacity() * size_of::<(u32, u32)>()
    }

    fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    fn len(&self) -> usize {
        self.keys.len()
    }

    fn emit(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        let keys = match emit_to {
            EmitTo::All => {
                self.map.clear();
                if let Some(wide) = &mut self.wide {
                    wide.clear();
                }
                mem::take(&mut self.keys)
            }
            EmitTo::First(n) => {
                let bits = self.bits;
                let mask = self.mask;
                self.map.retain(|value| {
                    if let Some(index) = ((*value >> bits) as usize).checked_sub(n) {
                        *value = (*value & mask) | ((index as u128) << bits);
                        true
                    } else {
                        false
                    }
                });
                if let Some(wide) = &mut self.wide {
                    wide.retain(|(_, index)| {
                        if let Some(remaining) = index.checked_sub(n) {
                            *index = remaining;
                            true
                        } else {
                            false
                        }
                    });
                }
                self.keys.drain(..n).collect()
            }
        };
        Ok(self.arrays(&keys))
    }

    fn clear_shrink(&mut self, num_rows: usize) {
        self.keys.clear();
        self.keys.shrink_to(num_rows);
        self.scratch.clear();
        self.scratch.shrink_to(num_rows);
        self.wide = None;
        self.map.clear();
        let (mask, random) = (self.mask, &self.random);
        self.map
            .shrink_to(num_rows, |key| ((key & mask) as i128).hash(random));
    }
}
