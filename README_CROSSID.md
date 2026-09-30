# CROSS-ID Phase 1 on CrossNews

This implementation completes the existing **training-free Phase 1** integration.
It uses frozen SELMA document embeddings and represents each author with a
normalized centroid, KMeans prototypes, and the full normalized reference bundle.
Scores combine top-k prototype cosine similarity, centroid cosine similarity,
and top-k reference cosine similarity. It does not implement the Phase 2 trainable
encoder or Phase 3 reranker described below.

For genre-aware ablations, normalized-centroid/reference baselines, and independent
silver validation, see [the ablation workflow](README_CROSSID_ABLATIONS.md).

## Setup

Use Python 3.10+ and install the CPU scoring dependencies:

```bash
python -m pip install -r requirements-crossid.txt
git lfs pull
python -m zipfile -e raw_data.zip .
python src/dataset_creation.py
```

The gold attribution protocol has 500 candidate authors, 15,000 reference
documents per reference condition (Article, Tweet, Both), and 15,000 held-out
targets (7,500 articles and 7,500 tweets). Both references contain 15 articles and
15 tweets per author; the single-genre conditions contain 30 per author.

## Required embeddings

Provide existing SELMA JSON files mapping document IDs to vectors:

- `selma_embeddings/mistral/train.json`: unprompted reference embeddings;
- `selma_embeddings/mistral/test_prompt_taskonly.json`: prompted target embeddings.

To generate them, separately install a compatible GPU-enabled PyTorch and
`sentence-transformers` environment. The existing generator loads
`intfloat/e5-mistral-7b-instruct` on CUDA (its default batch size is 15, documented
for an A40). Run from the repository root:

```bash
python src/generate_selma_embeddings.py train
python src/generate_selma_embeddings.py train combine
python src/generate_selma_embeddings.py test test_prompt_taskonly
python src/generate_selma_embeddings.py test test_prompt_taskonly combine
```

The scoring requirements do not install the embedding-generation stack. A CPU
scoring run needs no GPU once embeddings exist. Keep document IDs and splits
consistent with the CSVs; the benchmark rejects missing IDs, invalid vectors,
conflicting duplicate embeddings, and reference/target ID overlap.

## Reproducible benchmark

```bash
python src/benchmark_crossid.py --check
python src/benchmark_crossid.py --output results/crossid_benchmark
```

Use `--train-embeddings`, `--test-embeddings`, and `--data-dir` to override inputs.
Use a fresh output directory for each run. The runner evaluates upstream SELMA,
CROSS-ID `default`, and CROSS-ID `prototype_heavy` independently for all three
reference conditions, with Overall/Article/Tweet target breakdowns. It never
selects a configuration using test labels.

Outputs include:

- `summary.csv` and `summary.json`: Accuracy, R@8/16/32/64, reciprocal rank,
  mean/median rank, sample counts, and elapsed seconds per model/reference run;
- per-run predictions with author ordering and scores;
- `manifest.json`: input/source SHA-256 hashes, commit, package versions,
  configuration, split counts, tie policy, and completion status.

`Mean_Reciprical_Rank` retains the upstream metric key's spelling. Timings include
model loading, profile building, and evaluation, but exclude embedding generation
and output serialization. They are not pure inference latency.

SELMA uses the upstream raw-vector mean and negative cosine distance rounded to
four decimals. CROSS-ID normalizes each reference before aggregation. Both use
the same documents and embedding files. Ties are broken by ascending author ID,
where IDs come from sorted author labels. This fixes upstream optimistic tie
ranking, so compare against the SELMA result produced by this runner rather than
assuming exact parity with older published metrics.

## Original attribution CLI

```bash
python src/run_attribution.py \
  --model crossid --train --test \
  --query_file attribution_data/query/CrossNews_Article.csv \
  --target_file attribution_data/test/CrossNews.csv \
  --parameter_sets default prototype_heavy \
  --save_folder results
```

CROSS-ID configurations are saved separately under `results/<parameter-set>/`.
Both are tested independently; this path bypasses the upstream attribution
runner's nonexistent validation split. `--load --load_folder <saved-model-folder>`
restores `crossid_config.json`; reference profiles are rebuilt from the supplied
query CSV and embeddings. Run from the same working directory when stored paths
are relative.

## Validation and benchmark status

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

Tests use **synthetic fixtures only**, covering scoring, KMeans determinism,
profile-cache invalidation, tie ranking, invalid inputs, save/load, the
multi-configuration CLI, and the full benchmark output pipeline. Synthetic
accuracy is not evidence of CrossNews performance.

The first completed gold benchmark is committed under `results/crossid_retry`.
Its 500-author, 15,000-target run shows small top-1 gains for `prototype_heavy`
over SELMA, with mixed changes in ranking metrics and target genres. The
initial environment's missing-embedding record remains in
`CROSSID_BENCHMARK_STATUS.md` for provenance. New ablation scores require the
embedding files on the GPU machine; none are inferred from synthetic tests.

## Future phases (not implemented)

Phase 2: train on the silver split with cross-genre positive pairs, same-topic
hard negatives, topic/genre gradient-reversal heads, variable-size author
bundles, and learned prototypes. Phase 3: multi-prototype retrieval followed by
candidate-conditioned cross-encoder reranking and calibrated stylometry fusion.
These require a specified training/validation protocol and separate empirical
evaluation; they are not represented by the Phase 1 results.
