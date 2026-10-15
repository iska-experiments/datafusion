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
use arrow::array::{StringViewArray, UInt32Array};
use arrow::compute::take;
use arrow::datatypes::StringViewType;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::collections::HashSet;

#[test]
fn reused_views_preserve_values_across_arrays_and_prefix_emission() {
    for seed in 0..16 {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut builder = ByteViewGroupValueBuilder::<StringViewType>::new()
            .with_max_block_size(64)
            .with_view_reuse(true);
        let mut model = Vec::new();
        let mut held = Vec::new();
        for phase in 0..5 {
            let dictionary = StringViewArray::from(
                (0..32)
                    .map(|i| {
                        (i % 7 != 0).then(|| {
                            format!(
                                "same{}{}",
                                "ø🙂".repeat(i % 5),
                                char::from(b'a' + phase)
                            )
                        })
                    })
                    .collect::<Vec<_>>(),
            );
            let indices = UInt32Array::from(
                (0..400)
                    .map(|i| (i % 19 != 0).then(|| rng.random_range(0..32)))
                    .collect::<Vec<_>>(),
            );
            let input: ArrayRef = take(&dictionary, &indices, None).unwrap();
            let typed = input.as_string_view();
            let mut copied = HashSet::new();
            let expected = (0..typed.len())
                .filter(|&i| !typed.is_null(i) && typed.value(i).len() > 12)
                .filter(|&i| copied.insert(typed.views()[i]))
                .map(|i| typed.value(i).len())
                .sum::<usize>();
            let bytes = |builder: &ByteViewGroupValueBuilder<StringViewType>| {
                builder.in_progress.len()
                    + builder.completed.iter().map(Buffer::len).sum::<usize>()
            };
            let before = bytes(&builder);
            for row in 0..100 {
                builder.append_val(&input, row).unwrap();
            }
            builder
                .vectorized_append(&input, &(100..400).collect::<Vec<_>>())
                .unwrap();
            assert_eq!(Arc::strong_count(&input), 1, "Do not retain input buffers");
            assert_eq!(
                bytes(&builder) - before,
                expected,
                "Dictionary view copy work guard: seed{seed}/phase{phase}"
            );
            model.extend(typed.iter().map(|v| v.map(String::from)));
            let input_identity = Arc::downgrade(&input);
            drop(input);
            assert!(input_identity.upgrade().is_none());
            if phase % 2 == 0 {
                continue;
            }
            let emitted = rng.random_range(0..model.len());
            let output = builder.take_n(emitted);
            let expected = model.drain(..emitted).collect::<Vec<_>>();
            assert_eq!(
                output
                    .as_string_view()
                    .iter()
                    .map(|v| v.map(String::from))
                    .collect::<Vec<_>>(),
                expected
            );
            held.push((output, expected));
            for (row, value) in model.iter().enumerate() {
                assert_eq!(builder.nulls.is_null(row), value.is_none());
            }
        }
        let output = Box::new(builder).build();
        assert_eq!(
            output
                .as_string_view()
                .iter()
                .map(|v| v.map(String::from))
                .collect::<Vec<_>>(),
            model
        );
        for (output, expected) in held {
            output.to_data().validate_full().unwrap();
            assert_eq!(
                output
                    .as_string_view()
                    .iter()
                    .map(|v| v.map(String::from))
                    .collect::<Vec<_>>(),
                expected
            );
        }
    }
}
