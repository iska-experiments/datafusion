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
use arrow::array::{RecordBatch, StringArray, StringViewArray};
use datafusion_common::Result;
use datafusion_common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion_physical_expr_adapter::{PhysicalExprAdapter, PhysicalExprAdapterFactory};
use futures::TryStreamExt;

#[derive(Debug)]
struct Renamed;

#[derive(Debug)]
struct Names(Arc<dyn PhysicalExprAdapter>, SchemaRef);

impl PhysicalExprAdapter for Names {
    fn rewrite(&self, expr: Arc<dyn PhysicalExpr>) -> Result<Arc<dyn PhysicalExpr>> {
        self.0
            .rewrite(expr)?
            .transform(|e| {
                let Some(c) = e.as_any().downcast_ref::<Column>() else {
                    return Ok(Transformed::no(e));
                };
                Ok(Transformed::yes(Arc::new(Column::new(
                    self.1.field(c.index()).name(),
                    c.index(),
                )) as _))
            })
            .data()
    }
}

impl PhysicalExprAdapterFactory for Renamed {
    fn create(
        &self,
        logical: SchemaRef,
        physical: SchemaRef,
    ) -> Result<Arc<dyn PhysicalExprAdapter>> {
        let fields: Vec<_> = physical
            .fields()
            .iter()
            .enumerate()
            .map(|(i, f)| f.as_ref().clone().with_name(logical.field(i).name()))
            .collect();
        let adapter = DefaultPhysicalExprAdapterFactory
            .create(logical, Arc::new(Schema::new(fields)))?;
        Ok(Arc::new(Names(adapter, physical)))
    }
}

#[tokio::test]
async fn forced_views_preserve_values_without_expanding_renamed_dictionaries()
-> Result<()> {
    for seed in 0..8 {
        let dictionary: Vec<_> = [0, 1, 12, 13, 63, 64, 65, 1024]
            .into_iter()
            .map(|n| "ø🙂".repeat(n))
            .collect();
        let values: Vec<_> = (0..1024)
            .map(|i| {
                ((i + seed) % 13 != 0).then(|| dictionary[(i * 7 + seed) % 8].as_str())
            })
            .collect();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "stored",
            DataType::Utf8,
            true,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(StringArray::from(values.clone()))],
        )?;
        let mut bytes = Vec::new();
        let mut writer = ArrowWriter::try_new(&mut bytes, schema, None)?;
        writer.write(&batch)?;
        writer.close()?;
        let file = PartitionedFile::new("views.parquet", bytes.len() as u64);
        let store = Arc::new(InMemory::new());
        store.put(&file.object_meta.location, bytes.into()).await?;
        let logical = Arc::new(Schema::new(vec![Field::new(
            "renamed",
            DataType::Utf8View,
            true,
        )]));
        let expanded: usize = values.iter().flatten().map(|t| t.len()).sum();
        for enabled in [false, true] {
            let mut opener = ParquetOpenerBuilder::new()
                .with_store(store.clone())
                .with_schema(logical.clone())
                .with_projection_indices(&[0])
                .build();
            opener.schema_force_view_types = enabled;
            opener.expr_adapter_factory = Arc::new(Renamed);
            let batches: Vec<_> = opener.open(file.clone())?.await?.try_collect().await?;
            let mut actual = Vec::new();
            let mut payload = 0;
            for b in &batches {
                let strings = b
                    .column(0)
                    .as_any()
                    .downcast_ref::<StringViewArray>()
                    .unwrap();
                actual.extend(strings.iter());
                payload += strings
                    .data_buffers()
                    .iter()
                    .map(|b| b.len())
                    .sum::<usize>();
            }
            assert_eq!(actual, values, "seed {seed}, enabled {enabled}");
            assert_eq!(
                payload < expanded / 4,
                enabled,
                "seed {seed}: {payload} of {expanded}"
            );
        }
    }
    Ok(())
}
