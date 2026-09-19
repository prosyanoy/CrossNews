# CROSS-ID benchmark execution status

Status: **blocked on precomputed SELMA embeddings; no CrossNews performance results produced.**

Implementation base: `64ba26ca1c1ec56cb6dc5505b09d6098659a8f39`.

Completed:

- Downloaded `raw_data.zip` through Git LFS and extracted the gold/silver datasets.
- Ran the existing `python src/dataset_creation.py` attribution split generator.
- Passed 10 synthetic regression tests, including all 9 model/reference combinations in the benchmark runner.
- Ran the real-data benchmark preflight; it stopped before scoring because the two embedding files are absent.

| Split | Documents | Authors | Article | Tweet |
|---|---:|---:|---:|---:|
| attribution_data/query/CrossNews_Article.csv | 15000 | 500 | 15000 | 0 |
| attribution_data/query/CrossNews_Both.csv | 15000 | 500 | 7500 | 7500 |
| attribution_data/query/CrossNews_Tweet.csv | 15000 | 500 | 0 | 15000 |
| attribution_data/test/CrossNews.csv | 15000 | 500 | 7500 | 7500 |

Verified unique IDs, matching query/target author sets, and no query/target ID overlap for all three reference conditions.

Required files:

- `selma_embeddings/mistral/train.json`
- `selma_embeddings/mistral/test_prompt_taskonly.json`

The execution environment has no CUDA GPU or PyTorch/transformers installation. The upstream embedding generator requires CUDA for `intfloat/e5-mistral-7b-instruct`. No substitute encoder was used.

Resume after supplying matching embeddings:

```bash
python src/benchmark_crossid.py --check
python src/benchmark_crossid.py --output results/crossid_benchmark
```

Generated CSV SHA-256 hashes (row order may differ when regenerating the Both split because upstream iterates an author set):

```
64f332f8df55e0935f3ae9c63adfe1556dbd306bde475e491eefa5c63ac23fb1  attribution_data/query/CrossNews_Article.csv
b59bb95af1710dcf51d6298ca552c122e7b1a126ecbdfd40404e67b7e9902152  attribution_data/query/CrossNews_Both.csv
9172e9f4f4b918c4b16aac818f2f6a41670abedcd49ad8f44167e2f189fc0010  attribution_data/query/CrossNews_Tweet.csv
420953942952456c1ad6ecdc58800100219498c5174107c45ba58c64636aa169  attribution_data/test/CrossNews.csv
```

The implemented model remains Phase 1. Trainable topic/genre invariance and reranking are future phases, not completed or evaluated here.
