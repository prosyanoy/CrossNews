# CROSS-ID starter for CROSSNEWS

This is Phase 1 of the architecture discussed in chat.

## What is implemented

A drop-in attribution model compatible with the original CROSSNEWS
`AttributionModel` interface.

Instead of SELMA's single mean embedding per author, CROSS-ID builds:

1. a normalized author centroid;
2. KMeans prototypes over the author's reference-document embeddings;
3. the full normalized reference bundle.

For each target document, the score for an author is a weighted mixture of:

- top-k prototype cosine similarity;
- centroid cosine similarity;
- top-k direct reference cosine similarity.

This is the first implementation of the "aggregation-aware multi-prototype
author profile" idea. It is deliberately training-free and reuses existing
SELMA embeddings, so it can be benchmarked before adding the more expensive
topic/genre-invariant encoder.

## Install into a CrossNews checkout

Copy:

- `src/attribution_models/crossid.py`
- `src/model_parameters/crossid.json`

Then apply `run_attribution.patch` (or manually add the `crossid` branch).

## Prerequisite

Generate the SELMA embeddings first, following the upstream README and
`src/generate_selma_embeddings.py`.

Default configuration expects:

- `selma_embeddings/mistral/train.json`
- `selma_embeddings/mistral/test_prompt_taskonly.json`

If you use another prompt file, edit `src/model_parameters/crossid.json`.

## Run

```bash
python src/run_attribution.py \
  --model crossid \
  --train \
  --query_file attribution_data/query/CrossNews_Article.csv \
  --parameter_sets default prototype_heavy \
  --save_folder results \
  --test \
  --target_file attribution_data/test/CrossNews.csv
```

Repeat with `CrossNews_Tweet.csv` and `CrossNews_Both.csv`.

## Phase 2

The next implementation step should replace the frozen SELMA document
representation with a trainable encoder on the silver split:

- cross-genre positive pairs: Article(author u) <-> Tweet(author u)
- same-topic hard negatives from different authors
- gradient-reversal topic head
- gradient-reversal genre head
- variable-size author bundle/set encoder
- several learned author prototypes rather than KMeans prototypes

## Phase 3

Add a retrieval -> reranking pipeline:

1. fast multi-prototype retrieval over all authors;
2. retain top-N candidates;
3. candidate-conditioned cross-encoder over query/reference pairs;
4. calibrated fusion with low-level stylometry.

This keeps the benchmark closed-world and avoids web/RAG identity leakage.
