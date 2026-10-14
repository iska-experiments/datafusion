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
use arrow::array::{AsArray, Int64Array, RecordBatch};
use arrow::datatypes::{Field, Int64Type, Schema};
use std::{collections::BTreeMap, sync::Arc};

/// Match scalar key identities through growth, duplicates and forced collisions.
#[test]
fn entry_groups_preserve_scalar_identities() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, true),
        Field::new("b", DataType::Int64, true),
    ]));
    for seed in 0..8 {
        for buckets in [1, 7, 1024] {
            let mut values =
                GroupValuesColumn::<false>::try_new(Arc::clone(&schema)).unwrap();
            let empty: Vec<ArrayRef> = (0..2)
                .map(|_| Arc::new(Int64Array::from(Vec::<i64>::new())) as ArrayRef)
                .collect();
            values.intern(&empty, &mut Vec::new()).unwrap();
            let mut model = BTreeMap::new();
            let mut expected = BTreeMap::new();
            for phase in 0..8 {
                let rows = (0..97)
                    .map(|i| {
                        let i = (i * 7919 + phase * 13 + seed * 23) % 503;
                        let a = (i % 11 != 0).then_some(match i % 17 {
                            0 => i64::MIN,
                            1 => i64::MAX,
                            _ => i as i64 - 250,
                        });
                        let b = (i % 7 != 0).then_some((i % 37) as i64 - 18);
                        (a, b)
                    })
                    .collect::<Vec<_>>();
                let cols: Vec<ArrayRef> = vec![
                    Arc::new(Int64Array::from(
                        rows.iter().map(|r| r.0).collect::<Vec<_>>(),
                    )),
                    Arc::new(Int64Array::from(
                        rows.iter().map(|r| r.1).collect::<Vec<_>>(),
                    )),
                ];
                let hashes = rows
                    .iter()
                    .map(|(a, b)| {
                        (a.unwrap_or(0) as u64 ^ (b.unwrap_or(0) as u64).rotate_left(31))
                            % buckets
                    })
                    .collect::<Vec<_>>();
                let mut groups = vec![usize::MAX; rows.len()];
                values.collect_vectorized_process_context(&hashes, &mut groups);
                values.vectorized_append(&cols).unwrap();
                values.vectorized_equal_to(&cols, &mut groups);
                values
                    .scalarized_intern_remaining(&cols, &hashes, &mut groups)
                    .unwrap();
                for (row, key) in rows.iter().enumerate() {
                    if let Some(group) = model.insert(*key, groups[row]) {
                        assert_eq!(groups[row], group, "{seed}/{buckets}/{phase}/{row}");
                    }
                    if let Some(previous) = expected.insert(groups[row], *key) {
                        assert_eq!(previous, *key, "Distinct keys share a group");
                    }
                }
                assert!(
                    values.map_size
                        >= values.map.len() * size_of::<(u64, GroupIndexView)>()
                );
            }
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                values.emit(EmitTo::All).unwrap(),
            )
            .unwrap();
            assert_eq!(batch.num_rows(), expected.len());
            let a = batch.column(0).as_primitive::<Int64Type>();
            let b = batch.column(1).as_primitive::<Int64Type>();
            let actual = a.iter().zip(b.iter()).collect::<Vec<_>>();
            assert_eq!(actual, expected.into_values().collect::<Vec<_>>());
        }
    }
}
