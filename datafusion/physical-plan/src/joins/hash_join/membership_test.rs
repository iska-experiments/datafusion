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
use crate::common;
use crate::test::TestMemoryExec;
use arrow::array::Int64Array;
use arrow::compute::cast;
use arrow::datatypes::Field;
use datafusion_common::ScalarValue;
use datafusion_execution::config::SessionConfig;
use datafusion_physical_expr::expressions::Literal;
use rand::{Rng, SeedableRng, rngs::StdRng};

#[tokio::test]
async fn membership_matches_scalar_nullable_semantics() -> Result<()> {
    for seed in 0..4 {
        let mut rng = StdRng::seed_from_u64(seed);
        for ty in [
            DataType::Int8,
            DataType::Int16,
            DataType::Int32,
            DataType::Int64,
            DataType::UInt8,
            DataType::UInt16,
            DataType::UInt32,
            DataType::UInt64,
        ] {
            let base = (0..32)
                .filter(|_| rng.random_bool(0.6))
                .map(Some)
                .collect::<Vec<_>>();
            let probe = (0..137)
                .map(|_| rng.random_bool(0.8).then(|| rng.random_range(0..40)))
                .collect::<Vec<_>>();
            let schema = Arc::new(Schema::new(vec![
                Field::new("key", ty.clone(), true),
                Field::new("id", DataType::Int64, false),
            ]));
            let table = |keys: &[Option<i64>]| -> Result<Arc<dyn ExecutionPlan>> {
                let array = cast(&Int64Array::from(keys.to_vec()), &ty)?;
                let batch = RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![
                        array,
                        Arc::new(Int64Array::from_iter_values(0..keys.len() as i64)),
                    ],
                )?;
                let middle = keys.len() / 2;
                Ok(TestMemoryExec::try_new_exec(
                    &[vec![
                        RecordBatch::new_empty(Arc::clone(&schema)),
                        batch.slice(0, middle),
                        batch.slice(middle, keys.len() - middle),
                    ]],
                    Arc::clone(&schema),
                    None,
                )?)
            };
            for mode in 0..7 {
                if mode == 6 && ty != DataType::Int64 {
                    continue;
                }
                let mut build = base.clone();
                if mode == 1 {
                    build.extend(base.iter().copied());
                }
                build.extend([None, None]);
                if mode == 4 {
                    build.clear();
                }
                if mode == 6 {
                    build = vec![Some(i64::MIN), Some(i64::MAX), None];
                }
                let nulls = if mode == 2 {
                    NullEquality::NullEqualsNull
                } else {
                    NullEquality::NullEqualsNothing
                };
                let filter = (mode == 3).then(|| {
                    JoinFilter::new(
                        Arc::new(Literal::new(ScalarValue::Boolean(Some(false)))),
                        vec![],
                        Arc::new(Schema::empty()),
                    )
                });
                for join_type in [JoinType::RightSemi, JoinType::RightAnti] {
                    for projection in [vec![1], vec![]] {
                        let mut config = SessionConfig::default().with_batch_size(64);
                        config
                            .options_mut()
                            .execution
                            .perfect_hash_join_small_build_threshold = 819200;
                        let ctx =
                            Arc::new(TaskContext::default().with_session_config(config));
                        let join = HashJoinExec::try_new(
                            table(&build)?,
                            table(&probe)?,
                            vec![(
                                Arc::new(Column::new("key", 0)),
                                Arc::new(Column::new("key", 0)),
                            )],
                            filter.clone(),
                            &join_type,
                            Some(projection.clone()),
                            PartitionMode::CollectLeft,
                            nulls,
                            false,
                        )?;
                        let limit = (mode == 5).then_some(5);
                        let plan = join.with_fetch(limit).unwrap();
                        let batches = common::collect(plan.execute(0, ctx)?).await?;
                        let expected = probe
                            .iter()
                            .enumerate()
                            .filter_map(|(i, key)| {
                                let matched = mode != 3
                                    && build.iter().any(|other| match (key, other) {
                                        (Some(a), Some(b)) => a == b,
                                        (None, None) => {
                                            nulls == NullEquality::NullEqualsNull
                                        }
                                        _ => false,
                                    });
                                (matched == (join_type == JoinType::RightSemi))
                                    .then_some(i as i64)
                            })
                            .take(limit.unwrap_or(usize::MAX))
                            .collect::<Vec<_>>();
                        assert_eq!(
                            batches.iter().map(RecordBatch::num_rows).sum::<usize>(),
                            expected.len()
                        );
                        if !projection.is_empty() {
                            let actual = batches
                                .iter()
                                .flat_map(|batch| {
                                    batch
                                        .column(0)
                                        .as_any()
                                        .downcast_ref::<Int64Array>()
                                        .unwrap()
                                        .values()
                                        .to_vec()
                                })
                                .collect::<Vec<_>>();
                            assert_eq!(actual, expected);
                        }
                        let metrics = plan.metrics().unwrap();
                        let work = metrics
                            .sum_by_name("membership_probe_rows")
                            .map_or(0, |v| v.as_usize());
                        if mode == 0 {
                            assert_eq!(work, probe.len());
                        } else if mode != 5 {
                            assert_eq!(work, 0);
                        }
                    }
                }
            }
        }
    }
    Ok(())
}
