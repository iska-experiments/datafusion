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

use super::*;
use arrow::array::RecordBatch;
use arrow::datatypes::{Field, Schema};
use datafusion_common::{HashMap, ScalarValue};
use rand::{Rng, SeedableRng, rngs::StdRng};

#[test]
fn packed_keys_preserve_groups_through_emit_and_promotion() {
    for seed in 0..16 {
        let mut rng = StdRng::seed_from_u64(seed);
        for types in [
            vec![DataType::Int64, DataType::Int32],
            vec![DataType::UInt64, DataType::Int16],
            vec![DataType::Int32, DataType::Int32, DataType::Int32],
            vec![DataType::Int8, DataType::UInt8],
            vec![DataType::UInt32, DataType::UInt16, DataType::Int8],
            vec![DataType::Int8; 8],
        ] {
            let schema = Arc::new(Schema::new(
                types
                    .iter()
                    .enumerate()
                    .map(|(i, ty)| Field::new(i.to_string(), ty.clone(), true))
                    .collect::<Vec<_>>(),
            ));
            let mut packed = PackedGroups::try_new(Arc::clone(&schema)).unwrap();
            if seed % 2 == 0 {
                packed.limit = 8;
            }
            let mut model = HashMap::<Vec<ScalarValue>, usize>::new();
            let mut ordered = Vec::<Vec<ScalarValue>>::new();
            let mut groups = Vec::new();
            for round in 0..5 {
                let columns = types
                    .iter()
                    .map(|ty| {
                        let values = (0..257).map(|i| {
                            let n = match i % 17 {
                                0 => i64::MIN,
                                1 => i64::MAX,
                                _ => rng.random_range(-32..=32),
                            };
                            let n = (i % 11 != 0).then_some(n);
                            match ty {
                                DataType::Int8 => ScalarValue::Int8(n.map(|v| v as i8)),
                                DataType::UInt8 => ScalarValue::UInt8(n.map(|v| v as u8)),
                                DataType::Int16 => {
                                    ScalarValue::Int16(n.map(|v| v as i16))
                                }
                                DataType::Int32 => {
                                    ScalarValue::Int32(n.map(|v| v as i32))
                                }
                                DataType::Int64 => ScalarValue::Int64(n),
                                DataType::UInt16 => {
                                    ScalarValue::UInt16(n.map(|v| v as u16))
                                }
                                DataType::UInt32 => {
                                    ScalarValue::UInt32(n.map(|v| v as u32))
                                }
                                DataType::UInt64 => {
                                    ScalarValue::UInt64(n.map(|v| v as u64))
                                }
                                _ => unreachable!(),
                            }
                        });
                        ScalarValue::iter_to_array(values).unwrap().slice(1, 255)
                    })
                    .collect::<Vec<_>>();
                packed.intern(&columns, &mut groups).unwrap();
                for (row, &group) in groups.iter().enumerate() {
                    let key = columns
                        .iter()
                        .map(|a| ScalarValue::try_from_array(a, row).unwrap())
                        .collect::<Vec<_>>();
                    let expected = *model.entry(key.clone()).or_insert_with(|| {
                        let next = ordered.len();
                        ordered.push(key);
                        next
                    });
                    assert_eq!(group, expected);
                }
                assert_eq!(packed.len(), ordered.len());
                let count = if round == 4 {
                    ordered.len()
                } else {
                    ordered.len() / 3
                };
                let actual = packed
                    .emit(if round == 4 {
                        EmitTo::All
                    } else {
                        EmitTo::First(count)
                    })
                    .unwrap();
                let actual = RecordBatch::try_new(Arc::clone(&schema), actual).unwrap();
                for (row, expected) in ordered[..count].iter().enumerate() {
                    for (col, value) in expected.iter().enumerate() {
                        assert_eq!(
                            &ScalarValue::try_from_array(actual.column(col), row)
                                .unwrap(),
                            value
                        );
                    }
                }
                ordered.drain(..count);
                model = ordered
                    .iter()
                    .cloned()
                    .enumerate()
                    .map(|(id, key)| (key, id))
                    .collect();
                assert_eq!(packed.len(), ordered.len());
            }
            assert!(packed.is_empty());
            packed.clear_shrink(4);
            assert!(packed.wide.is_none());
            assert!(packed.size() < 4096);
        }
    }
}
