# CROSS-ID Phases 2 and 3: first trainable implementation

This adds runnable silver training and frozen evaluation workflows. **No real
Phase 2/3 accuracy is claimed yet.** CPU tests exercise both optimizers with
synthetic embeddings and a tiny locally initialized Transformer; they establish
that the workflow runs, not that it improves author attribution.

Phase 2 starts with a trainable residual adapter over the existing SELMA
embeddings. The 7B SELMA backbone remains frozen; this is not end-to-end SELMA
fine-tuning. It implements cross-genre positives, same-pseudo-topic negative
author sampling, gradient-reversal topic and genre heads, variable-size author
bundles, and learned attention prototypes shared across authors. Novel authors
receive profiles from their reference documents, without train-identity slots.
Backbone fine-tuning can follow after this cheaper milestone is measured.

Phase 3 retrieves a fixed candidate shortlist with the Phase 2 model, scores
query/reference text pairs with a trained cross-encoder, and averages pair
logits per candidate. Logistic fusion combines retrieval, local reference,
centroid, cross-encoder, and low-level stylometry distance features. It is fitted
only on a separate silver calibration author split. No author labels, usernames
from CSV metadata, or external web/identity retrieval enter the text model.
Raw source texts are not anonymized; this is not a content-redacted benchmark.

## Data protocol

Run from the repository root:

```bash
# Install the correct CUDA PyTorch build for your machine first.
python -m pip install -r requirements-crossid-learning.txt
# Needed only for generating the frozen SELMA embedding views:
python -m pip install sentence-transformers
git lfs pull
python -m zipfile -e raw_data.zip .
python src/prepare_crossid_learning.py
```

Preparation excludes every gold author, document ID, and normalized text. A
disk-backed SQLite index keeps one canonical document ID per casefolded,
whitespace-normalized silver text, preventing duplicate texts from crossing
roles. Author/document selection is seeded and independent of input JSON order.
The output is a fresh, hashed `crossid_learning_data/` directory.

Default preparation executed on the provided archive gives:

| Role | Authors | Use |
|---|---:|---|
| train | 117 | adapter and cross-encoder optimization |
| dev | 60 | checkpoint selection |
| calibration | 60 | logistic fusion fitting |
| test | 60 | frozen independent silver evaluation |

After deduplication, 297 non-gold authors have at least 45 documents per genre.
Training keeps up to 100 documents per author per genre (21,608 in this run).
Each reference condition has 30 documents per author: Both uses 15 articles and
15 tweets; Article/Tweet uses 30 of that genre. Targets have 15 documents per
genre per author. Each held-out role therefore has 1,800 references per condition
and 1,800 targets. The full reference/target embedding views contain 32,408 and
27,008 documents respectively, including both views of every training document.

This is a new author partition. The earlier 300-author Phase 1 silver validation
set is not an independent holdout for these models: some of its authors now
train the adapter. Gold has already informed exploratory designs, so report
gold results as exploratory. The 60-author silver accuracy is not directly
comparable with the 500-author gold accuracy.

## Generate both embedding views

Use the same SELMA model, precision, prompt, and text truncation as the original
benchmark. The generator uses `intfloat/e5-mistral-7b-instruct`, the first 5,000
characters, and four-decimal embeddings. Training anchors use the prompted view;
positive/reference bundles use the unprompted view, matching target/reference
roles at inference. IDs intentionally overlap between the two training views.
They are validated separately rather than merged into one ID dictionary.

```bash
python src/generate_selma_embeddings.py train \
  --input-csv crossid_learning_data/embedding_references.csv \
  --output-dir selma_embeddings/crossid_learning --batch-size 8
python src/generate_selma_embeddings.py train combine \
  --output-dir selma_embeddings/crossid_learning
python src/generate_selma_embeddings.py test test_prompt_taskonly \
  --input-csv crossid_learning_data/embedding_targets.csv \
  --output-dir selma_embeddings/crossid_learning --batch-size 8
python src/generate_selma_embeddings.py test test_prompt_taskonly combine \
  --output-dir selma_embeddings/crossid_learning
```

Use a fresh embedding directory for this protocol; partition filenames do not
encode input hashes. Do not mix partitions from a different CSV or prompt.
The generator needs CUDA. Adapter training can run on CPU once embeddings exist;
the pretrained cross-encoder is much more practical on CUDA. Reduce embedding
or cross-encoder batch sizes if the GPU runs out of memory. No hardware-specific
VRAM measurement has been made for the new workflow.

## Train Phase 2

```bash
python src/train_crossid_phase2.py \
  --reference-embeddings selma_embeddings/crossid_learning/train.json \
  --target-embeddings selma_embeddings/crossid_learning/test_prompt_taskonly.json \
  --device cuda --output results/crossid_phase2
```

Add `--check` to validate inputs without training/writing outputs. Defaults are
a 256-dimensional adapter, four attention prototypes, 16 distinct authors per
batch, variable bundles of 1–8 opposite-genre references, ten epochs, and 200
steps per epoch. Pseudo topics are 32 TF-IDF MiniBatchKMeans content clusters
fitted only on training documents, not ground-truth topic annotations. Sampling
prefers different authors in the anchor's pseudo topic, then fills from other
authors when needed; not every negative is guaranteed to share a topic.

The loss combines bidirectional document contrastive classification, anchor-to-
bundle author classification, topic/genre gradient reversal, and a small
prototype diversity penalty. Dev retrieval uses a fixed 0.5 learned-prototype /
0.5 top-four-reference mixture; the best overall dev MRR selects `adapter.pt`.
Heads/slots are genuinely optimized. Adversarial training is an objective,
not evidence that learned representations have achieved topic/genre invariance.

## Train Phase 3, then fit fusion

```bash
python src/train_crossid_phase3.py train \
  --phase2 results/crossid_phase2/adapter.pt \
  --reference-embeddings selma_embeddings/crossid_learning/train.json \
  --target-embeddings selma_embeddings/crossid_learning/test_prompt_taskonly.json \
  --device cuda --output results/crossid_phase3

python src/train_crossid_phase3.py calibrate \
  --phase2 results/crossid_phase2/adapter.pt --phase3 results/crossid_phase3 \
  --reference-embeddings selma_embeddings/crossid_learning/train.json \
  --target-embeddings selma_embeddings/crossid_learning/test_prompt_taskonly.json \
  --device cuda --output results/crossid_phase3/fusion.json
```

The initial cross-encoder is `distilbert-base-uncased`; use `--model` for another
compatible Hugging Face sequence classifier or a local model. `--revision`
can pin the initial model revision; the resolved revision and saved model hashes
are recorded. Defaults use 20 candidates, three nearest references per candidate,
512 total pair tokens, 2,000 characters per text, batch size 16, three epochs,
and at most 100,000 sampled training pairs per epoch. Dev checkpoint selection
uses a fixed balanced subset with every candidate author represented (240
queries under the default 300-query cap and 60-author dev split).

The true author is added to a missed shortlist only when mining *training*
pairs. Dev, calibration, and test never receive this injection. Reranking cannot
recover a missed author; omitted candidates retain their retrieval order below
the shortlist. Full author ranks and Phase 2 candidate recall are reported.

Fusion uses standardized retrieval/local/centroid scores, averaged cross-encoder
logits, and absolute target-to-author-mean stylometry differences. Stylometry
includes character/word counts, word-length statistics, punctuation/case/digit/
whitespace ratios, and fixed function-word frequencies. StandardScaler and
L2 logistic regression (`C=1`) are fitted on calibration candidates only, then
stored as numeric JSON parameters. Probability calibration under a different
candidate population, such as gold, is not guaranteed. No test-based fitting
or hyperparameter selection occurs in these scripts.

## Benchmark frozen models

```bash
python src/benchmark_crossid_phase23.py \
  --data-dir crossid_learning_data/test \
  --phase2 results/crossid_phase2/adapter.pt --phase3 results/crossid_phase3 \
  --fusion results/crossid_phase3/fusion.json \
  --reference-embeddings selma_embeddings/crossid_learning/train.json \
  --target-embeddings selma_embeddings/crossid_learning/test_prompt_taskonly.json \
  --device cuda --output results/crossid_phase23_silver_test

# Exploratory gold transfer using your existing gold embedding files:
python src/benchmark_crossid_phase23.py \
  --data-dir attribution_data \
  --phase2 results/crossid_phase2/adapter.pt --phase3 results/crossid_phase3 \
  --fusion results/crossid_phase3/fusion.json \
  --reference-embeddings selma_embeddings/mistral/train.json \
  --target-embeddings selma_embeddings/mistral/test_prompt_taskonly.json \
  --allow-reference-text-overlap --device cuda --output results/crossid_phase23_gold
```

Use `--check` first. The initial fusion is calibrated for Both references and
the default 20/3 shortlist/reference budgets. Evaluation rejects different
budgets, changed adapter/cross-encoder files, changed inference source, or
overlapping training/dev/calibration authors. Without `--fusion`, Phase 2 and
the cross-encoder can also be evaluated with `--reference-sets Article Tweet Both`.
Without `--phase3`, the runner benchmarks Phase 2 and frozen baselines only.
Changing an inference implementation requires refitting its affected artifacts.

Outputs compare six stages on identical documents: SELMA, frozen
`reference_only`, adapted reference-only, learned profile retrieval, cross-encoder
reranking, and fused reranking. Baselines are vectorized implementations of the
Phase 1 scoring formulations. The runner saves Overall/Article/Tweet metrics,
compact `prediction_ranks.csv` files, candidate recall, and hashed manifests.
The upstream gold split contains 35/7/27 shared normalized texts between
Article/Tweet/Both references and targets. The explicit gold option above keeps
the original documents for comparability, records affected target counts, and
additionally saves metrics excluding targets with reference-text duplicates.
Reference/target ID overlap and training-author overlap are still rejected.
New silver preparation removes normalized duplicate texts before splitting.
Full ranks count shortlist misses; recall metrics do not silently drop them.
Wall runtime in the manifest excludes preflight and embedding generation, and
is not per-stage inference latency. Use a fresh result directory for every run.

## Local validation

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

Learning tests require the learning dependencies; the Phase 1 CPU-only suite
can still run without them, with learning tests skipped. They check reverse
gradients, actual optimizer updates to adapter/slots/adversarial heads, bundle
padding/permutation invariance, cross-genre sampling, order-independent data
selection, leakage/hash rejection, no evaluation oracle injection, full tail
ranks, baseline scoring, and an end-to-end tiny-Transformer training/calibration/
benchmark run. The real raw-data split preparation was executed and all twelve
reference/target combinations passed ID/text/author checks. Full pretrained
training and performance evaluation remain to be run on the GPU machine.
