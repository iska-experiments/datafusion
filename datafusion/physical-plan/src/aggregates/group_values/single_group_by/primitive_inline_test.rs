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
use arrow::array::{Int64Array, types::Int64Type};
use std::collections::HashMap;

#[test]
fn inline_integer_groups_match_scalar_model() {
    for seed in 0..16_u64 {
        let mut state = seed + 1;
        let mut groups = GroupValuesPrimitive::<Int64Type>::new(DataType::Int64);
        assert!(matches!(groups.map, KeyMap::Inline(_)), "Inline key guard");
        let mut expected = Vec::new();
        let mut positions = HashMap::new();
        for round in 0..8 {
            let values = (0..4096)
                .map(|i| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    match i % 29 {
                        0 => None,
                        1 => Some(i64::MIN),
                        2 => Some(i64::MAX),
                        _ => Some(if round % 2 == 0 {
                            state as i64
                        } else {
                            (state % 8192) as i64
                        }),
                    }
                })
                .collect::<Vec<_>>();
            let array = Arc::new(Int64Array::from(values.clone())) as ArrayRef;
            let mut actual = Vec::new();
            groups.intern(&[array], &mut actual).unwrap();
            let model = values
                .iter()
                .map(|v| {
                    *positions.entry(*v).or_insert_with(|| {
                        let index = expected.len();
                        expected.push(*v);
                        index
                    })
                })
                .collect::<Vec<_>>();
            assert_eq!(actual, model);
            assert_eq!(groups.len(), expected.len());
            assert!(!groups.is_empty());
            let emit = if round == 7 {
                expected.len()
            } else {
                expected.len() / 3
            };
            let arrays = groups.emit(EmitTo::First(emit)).unwrap();
            assert_eq!(
                arrays[0]
                    .as_primitive::<Int64Type>()
                    .iter()
                    .collect::<Vec<_>>(),
                expected[..emit]
            );
            expected.drain(..emit);
            positions = expected.iter().enumerate().map(|(i, v)| (*v, i)).collect();
        }
        assert!(groups.is_empty());
        groups.clear_shrink(16);
        assert_eq!(groups.len(), 0);
        let arrays = groups.emit(EmitTo::All).unwrap();
        assert_eq!(arrays[0].len(), 0);
        let large = GroupValuesPrimitive::<arrow::datatypes::Decimal128Type>::new(
            DataType::Decimal128(38, 0),
        );
        assert!(
            matches!(large.map, KeyMap::Hashed(_)),
            "Large key hash guard"
        );
    }
}
