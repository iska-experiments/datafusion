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

#[test]
fn retained_buffers_bound_memory_and_preserve_prefix_values() {
    for seed in 0..16 {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut builder = ByteViewGroupValueBuilder::<StringViewType>::new()
            .with_max_block_size(64)
            .with_buffer_reuse(true);
        let mut model = Vec::new();
        let mut held = Vec::new();
        for phase in 0..5 {
            let mut dictionary = StringViewArray::from(
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
            dictionary = StringViewArray::new(
                dictionary.views().clone(),
                dictionary
                    .data_buffers()
                    .iter()
                    .map(|b| Buffer::from_vec(b.to_vec()))
                    .collect::<Vec<_>>(),
                dictionary.nulls().cloned(),
            );
            if seed % 4 == 0 {
                let mut buffers = dictionary.data_buffers().to_vec();
                let mut padded = buffers[0].to_vec();
                padded.resize(16 << 20, b'x');
                buffers[0] = Buffer::from_vec(padded);
                dictionary = StringViewArray::new(
                    dictionary.views().clone(),
                    buffers,
                    dictionary.nulls().cloned(),
                );
            } else if seed % 4 == 1 {
                let mut buffers = dictionary.data_buffers().to_vec();
                buffers.extend((0..4096).map(|_| Buffer::from_vec(vec![0u8])));
                dictionary = StringViewArray::new(
                    dictionary.views().clone(),
                    buffers,
                    dictionary.nulls().cloned(),
                );
            } else if seed % 4 == 2 {
                let buffers: Vec<_> = dictionary
                    .data_buffers()
                    .iter()
                    .map(|b| {
                        let owner = Arc::new(b.to_vec());
                        let ptr =
                            std::ptr::NonNull::new(owner.as_ptr() as *mut u8).unwrap();
                        // SAFETY: owner keeps allocation alive and its bytes immutable
                        unsafe { Buffer::from_custom_allocation(ptr, owner.len(), owner) }
                    })
                    .collect();
                dictionary = StringViewArray::new(
                    dictionary.views().clone(),
                    buffers,
                    dictionary.nulls().cloned(),
                );
            }
            let rows = 1024 + seed as usize;
            let indices = UInt32Array::from(
                (0..rows)
                    .map(|i| (i % 19 != 0).then(|| rng.random_range(0..32)))
                    .collect::<Vec<_>>(),
            );
            let input: ArrayRef = take(&dictionary, &indices, None).unwrap();
            let typed = input.as_string_view();
            let payload = typed
                .iter()
                .map(|s| s.map_or(16, |s| s.len().max(16)))
                .sum::<usize>();
            let bytes = |b: &ByteViewGroupValueBuilder<StringViewType>| {
                b.in_progress.capacity()
                    + b.completed.capacity() * size_of::<Buffer>()
                    + b.completed
                        .iter()
                        .map(|b| b.capacity().max(b.len()))
                        .sum::<usize>()
            };
            let before = bytes(&builder);
            if phase % 2 == 0 {
                builder
                    .vectorized_append(&input, &(0..rows).collect::<Vec<_>>())
                    .unwrap();
                let source_bytes =
                    typed.data_buffers().iter().try_fold(0usize, |n, b| {
                        n.checked_add(b.capacity().max(b.len()) + 2 * size_of::<Buffer>())
                    });
                if source_bytes.is_some_and(|bytes| bytes <= rows * 16) {
                    let base =
                        builder.input_base.expect("Bounded buffer reuse work guard")
                            as usize;
                    for (i, buffer) in typed.data_buffers().iter().enumerate() {
                        assert_eq!(
                            builder.completed[base + i].as_ptr(),
                            buffer.as_ptr(),
                            "Reuse actual source buffer without payload copy"
                        );
                    }
                    assert!(builder.in_progress.is_empty());
                } else {
                    assert!(
                        builder.input_base.is_none(),
                        "Reject sparse retained allocation"
                    );
                }
            } else {
                for row in 0..rows {
                    builder.append_val(&input, row).unwrap();
                }
            }
            assert!(
                bytes(&builder) - before <= 4 * payload,
                "Retained allocation bound: seed{seed}/phase{phase}"
            );
            assert_eq!(Arc::strong_count(&input), 1, "Do not retain input array");
            model.extend(typed.iter().map(|s| s.map(String::from)));
            let identity = Arc::downgrade(&input);
            drop(input);
            assert!(identity.upgrade().is_none());
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
                    .map(|s| s.map(String::from))
                    .collect::<Vec<_>>(),
                expected
            );
            held.push((output, expected));
        }
        let output = Box::new(builder).build();
        assert_eq!(
            output
                .as_string_view()
                .iter()
                .map(|s| s.map(String::from))
                .collect::<Vec<_>>(),
            model
        );
        for (output, expected) in held {
            output.to_data().validate_full().unwrap();
            assert_eq!(
                output
                    .as_string_view()
                    .iter()
                    .map(|s| s.map(String::from))
                    .collect::<Vec<_>>(),
                expected
            );
        }
        let text = "ø🙂".repeat(4096 + seed as usize);
        let dictionary = StringViewArray::from(vec![Some(text.as_str())]);
        let dictionary = StringViewArray::new(
            dictionary.views().clone(),
            dictionary
                .data_buffers()
                .iter()
                .map(|b| Buffer::from_vec(b.to_vec()))
                .collect::<Vec<_>>(),
            dictionary.nulls().cloned(),
        );
        let rows = 4096 + seed as usize;
        let indices = UInt32Array::from(
            (0..rows)
                .map(|i| (i % 11 != 0).then_some(0))
                .collect::<Vec<_>>(),
        );
        let input = take(&dictionary, &indices, None).unwrap();
        let mut builder =
            ByteViewGroupValueBuilder::<StringViewType>::new().with_buffer_reuse(true);
        builder
            .vectorized_append(&input, &(0..rows).collect::<Vec<_>>())
            .unwrap();
        let allocation = |buffers: &[Buffer]| {
            buffers
                .iter()
                .map(|b| b.capacity().max(b.len()))
                .sum::<usize>()
        };
        let before = allocation(&builder.completed);
        assert!(
            builder.input_base.is_some(),
            "Retain dense long-string input"
        );
        drop(input);
        for emitted in [0, rows / 2] {
            let output = builder.take_n(emitted);
            assert!(
                allocation(output.as_string_view().data_buffers())
                    + allocation(&builder.completed)
                    <= 3 * before,
                "Prefix allocation work guard"
            );
            output.to_data().validate_full().unwrap();
        }
        let remaining = Box::new(builder).build();
        assert!(
            remaining
                .as_string_view()
                .iter()
                .flatten()
                .all(|v| v == text)
        );
    }
}
