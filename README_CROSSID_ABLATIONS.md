# CROSS-ID genre ablations and silver validation

The first gold run in `results/crossid_retry` showed small top-1 gains for
`prototype_heavy`, but mixed references improved article targets while hurting
tweet targets. The following fixed experiments separate prototype allocation,
aggregation, and observed-target-genre routing. They do not train a new encoder.

## Configurations

| Parameter set | What changes from the original pooled model |
|---|---|
| `default`, `prototype_heavy` | Original pooled controls, unchanged |
| `centroid_only` | Cosine similarity to the mean of individually normalized references |
| `reference_only` | Mean of the top 4 individual reference cosine similarities |
| `genre_prototypes`, `genre_prototypes_heavy` | Fit KMeans separately by reference genre; retain pooled centroid, references, and top-k scoring |
| `genre_balanced`, `genre_balanced_heavy` | Build genre-specific prototype/centroid/reference profiles and average their scores with equal genre weights |
| `genre_matched`, `genre_matched_heavy` | Score the profile matching the observed target genre; fall back to available genres when that genre has no references |

Default-weight variants keep the original 0.55/0.20/0.25 mixture and a total
prototype budget of 4. Heavy variants keep 0.70/0.10/0.20 and a budget of 6.
Genre-aware modes split the total budget equally across Article/Tweet (2+2 or
3+3), capped by documents available in each group. This does not double the
prototype budget. Single-genre conditions reduce to the pooled score, except
that centroid/reference-only ablations deliberately change the mixture.

`genre_prototypes` isolates prototype allocation. `genre_balanced` additionally
changes centroid/reference aggregation, so improvements in that variant alone
cannot be attributed solely to KMeans. Target-genre matching uses the observable
CSV `genre` field, never the target's author label. It requires genre metadata
and is not a genre-invariant encoder.

## Exploratory gold ablation using existing embeddings

Install updated CPU dependencies, then run from the repository root:

```bash
python -m pip install -r requirements-crossid.txt
python src/benchmark_crossid.py \
  --parameter-sets all --reference-sets Both \
  --output results/crossid_genre_ablation
```

This compares all ten fixed CROSS-ID variants against SELMA on the existing
mixed-reference gold data. Remove `--reference-sets Both` to cover Article,
Tweet, and Both; the genre-aware single-genre scores are redundant controls.
Use a new output directory for every run. Do not choose a configuration from
these gold results. Gold was already inspected while designing these variants,
so subsequent gold analyses remain exploratory even with independent silver
validation. A fresh test set would be needed for a confirmatory generalization
claim.

## Prepare independent silver validation

The provided archive contains 309 non-gold silver authors with at least 45
usable documents per genre. The default protocol samples 300 of them using seed
20260930, excluding every gold author and every gold document ID. It keeps 30
reference documents and 15 target documents per genre per author. Reference
conditions therefore have 9,000 documents each; the target set has 9,000
(4,500 articles, 4,500 tweets). Mixed references use 15 documents of each genre
per author. The reference-embedding union has 18,000 unique IDs.

```bash
python src/prepare_crossid_validation.py
```

This requires `raw_data/crossnews_gold.json` and `crossnews_silver.json`. It uses
two streaming passes and bounded per-author document heaps to avoid loading the
entire silver JSON into memory. It writes `crossid_validation_data/` with an
input-hashed `validation_protocol.json`. The same seed and document IDs produce
the same CSVs even when the raw JSON record order changes. It refuses to
overwrite an existing output directory. `--authors`, `--references-per-genre`,
`--targets-per-genre`, `--seed`, and `--output` are available for explicitly
separate protocols; the original gold splits are left intact.

The 300-author validation accuracy is not directly comparable with the
500-author gold accuracy. Compare configurations within each split.

## Generate validation embeddings once on the GPU machine

Use the same GPU environment and model precision as your original SELMA run.
The generator retains the original model, 5,000-character text clipping,
task-only target prompt, four-decimal output, and default batch size 15. Store
validation embeddings separately from gold embeddings:

```bash
python src/generate_selma_embeddings.py train \
  --data-dir crossid_validation_data --output-dir selma_embeddings/validation
python src/generate_selma_embeddings.py train combine \
  --output-dir selma_embeddings/validation
python src/generate_selma_embeddings.py test test_prompt_taskonly \
  --data-dir crossid_validation_data --target-split validation \
  --output-dir selma_embeddings/validation
python src/generate_selma_embeddings.py test test_prompt_taskonly combine \
  --output-dir selma_embeddings/validation
```

Adjust `--batch-size` if needed. Interrupted-generation partitions use the
upstream resumable layout; do not reuse a partition directory after changing
its dataset, model, or generation settings. Benchmark preflight verifies ID
coverage before any scoring. Existing gold embeddings cannot cover the new
silver IDs: 18,000 reference and 9,000 prompted target embeddings are needed.

## Freeze choices on validation, then evaluate gold

```bash
python src/benchmark_crossid.py \
  --data-dir crossid_validation_data --split validation \
  --train-embeddings selma_embeddings/validation/train.json \
  --test-embeddings selma_embeddings/validation/test_prompt_taskonly.json \
  --parameter-sets all --reference-sets Both \
  --output results/crossid_silver_validation

python src/select_crossid_validation.py \
  --validation-results results/crossid_silver_validation \
  --output results/crossid_silver_selection.json

python src/benchmark_crossid.py \
  --selection results/crossid_silver_selection.json --reference-sets Both \
  --output results/crossid_selected_gold
```

The predeclared primary selection metric is overall top-1 accuracy on the
balanced validation targets; ties use overall reciprocal rank, then parameter
set name. Selection chooses among CROSS-ID ablations, retaining SELMA as a
separate fixed baseline. To choose independently for all three reference
conditions, omit `--reference-sets Both` from both validation and selected-test
commands.

The selector rejects test manifests and incomplete validation runs. Validation
preflight requires a silver protocol, unchanged split hashes, balanced targets,
and disjoint gold authors. Selected-test preflight rejects overlapping
validation/test authors, altered selected configurations, or changed CROSS-ID
source. The frozen selection and its validation hashes are recorded in the test
manifest. A selection file is created exclusively and cannot be overwritten.

## Outputs and validation status

Each run retains the original summary and full predictions, and now also saves
`prediction_ranks.csv` per configuration/reference condition. These compact
files include ID, genre, true author, predicted author, and rank without all 500
candidate scores; commit them when sharing results for paired significance and
error analysis.

Validation preparation was executed successfully on the real archive and its
counts and gold exclusion were checked. Regression tests cover prototype
budgets, isolated pooled-centroid/reference preservation, equal genre weighting,
observed-genre routing and cross-genre fallback, cache invalidation on genre
changes, deterministic streaming splits, selection/test separation, frozen
configuration checks, compact predictions, and custom embedding output paths.

No new real ablation accuracy is reported by this change: the committed
repository contains the first run's metrics but not its embedding JSON files,
and the execution environment has no CUDA GPU to generate silver embeddings.
