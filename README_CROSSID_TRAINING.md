# CROSS-ID Phases 2 and 3: experimental training pipeline

Phase 2 trains a residual adapter and projection over **frozen SELMA embeddings**,
plus shared attention queries that build learned prototypes from an author's
reference bundle. This is an adapter experiment, not SELMA backbone fine-tuning.
Phase 3 trains a candidate-conditioned text cross-encoder, then fits stylometry
fusion on separate calibration authors. Both stages run independently of the
Phase 1 CLI and preserve its existing results.

No real Phase 2/3 accuracy is claimed. CPU tests exercise both training loops,
checkpoint restoration, calibration, leakage rejection, and end-to-end ranking
on synthetic data. An offline randomly initialized transformer is available as
`--encoder tiny` solely for wiring tests. Use a pretrained encoder for research.

## 1. Setup and prepare the protocol

Use Python 3.10+ in your existing GPU environment. Install a suitable PyTorch
build (`torch>=2.2`) separately, then:

```bash
git pull --ff-only origin codex/crossid-benchmark
python -m pip install -r requirements-crossid-training.txt
# Only needed if generating new SELMA embeddings:
python -m pip install sentence-transformers
git lfs pull --include=raw_data.zip
python -m zipfile -e raw_data.zip .
python src/prepare_crossid_training.py
```

Preparation streams silver, excludes every gold author and document ID, and
globally removes repeated text, including text present in gold. Text hashes
ignore case and whitespace. ID hashes select documents deterministically from
the surviving pool. The first occurrence of duplicate text is retained, so
the exact source file order is part of the hashed protocol.

The real preparation produced 297 eligible authors on the current data:

| Cohort | Authors | Unprompted reference documents* | Prompted targets |
|---|---:|---:|---:|
| Train | 117 | 7,020 | 3,510 |
| Development | 60 | 3,600 | 1,800 |
| Calibration | 60 | 3,600 | 1,800 |
| Test | 60 | 3,600 | 1,800 |
| Total | 297 | 17,820 | 8,910 |

*Each author has 30 Article and 30 Tweet references available for embedding
generation. Each reference condition uses 30 documents per author: 30 of one
genre or 15 of each for Both. All conditions share 15 targets per genre per
author. Training uses Both bundles. CSV and source hashes, cohort author lists,
seed, and exclusion counts are in `crossid_training_data/protocol.json`.
Preparation requires a fresh output directory and fails if too few authors
qualify. `--heldout-authors` changes the three equal held-out cohort sizes.

The four roles are disjoint in authors, IDs, and normalized text. Development
selects the best epoch. Calibration fits fusion once. Test evaluates frozen
choices. Do not use test outcomes to choose hyperparameters. The earlier
300-author Phase 1 validation sample overlaps this new protocol and must not
be substituted for any of these roles.

## 2. Generate silver embeddings

Your existing gold embeddings do not include these documents. Generate
unprompted references and prompted targets for each cohort with the existing
SELMA generator. Use the same backbone/prompt/precision as your gold embeddings.
The following Bash loop uses fresh cohort-specific directories:

```bash
for stage in train dev calibration test; do
  python src/generate_selma_embeddings.py train \
    --data-dir crossid_training_data/$stage \
    --output-dir selma_embeddings/crossid_training/$stage --batch-size 2
  python src/generate_selma_embeddings.py train combine \
    --output-dir selma_embeddings/crossid_training/$stage
  python src/generate_selma_embeddings.py test test_prompt_taskonly \
    --data-dir crossid_training_data/$stage \
    --output-dir selma_embeddings/crossid_training/$stage --batch-size 2
  python src/generate_selma_embeddings.py test test_prompt_taskonly combine \
    --output-dir selma_embeddings/crossid_training/$stage
done
```

The generator still uses its existing CUDA backbone loading. A smaller batch
reduces activation memory but does not reduce model-weight memory. Its legacy
partition files are tied to input order and batch size: use fresh output paths,
and do not reuse partitions after changing the split or batching. The trained
pipeline validates all required IDs and vector dimensions before using them.

## 3. Train Phase 2

```bash
python src/crossid_trained.py train-adapter \
  --data-dir crossid_training_data --device cuda \
  --train-embeddings selma_embeddings/crossid_training/train/train.json \
  --target-embeddings selma_embeddings/crossid_training/train/test_prompt_taskonly.json \
  --dev-train-embeddings selma_embeddings/crossid_training/dev/train.json \
  --dev-target-embeddings selma_embeddings/crossid_training/dev/test_prompt_taskonly.json \
  --output crossid_checkpoints/phase2
```

The fixed defaults are 5 epochs, 200 optimizer steps per epoch, 8 authors per
batch, a 256-dimensional output, and 4 learned attention prototypes. Every
batch samples variable reference counts per author and genre, and one prompted
target per genre. Objectives are:

- Supervised contrastive loss with cross-genre same-author positives.
- Additional pressure on same-topic in-batch negatives via a fixed logit bias.
  Topics are train-only TF-IDF/KMeans proxies, not annotated topic labels.
- Author-bundle classification using a smooth maximum over learned prototypes.
- Topic and genre heads with gradient reversal; inference discards the heads.
- A small penalty for highly similar prototypes.

Prototype queries are global parameters, not a table of training-author IDs.
Held-out author prototypes are computed from their references only. Retrieval
uses an equal mixture of smooth prototype matching and top-4 individual
reference cosine matching. `adapter.pt` and `metadata.json` record the earliest
epoch with highest development top-1, hashes, identities, objectives, settings,
and history. Set `--device cpu` for adapter training if needed.

## 4. Train Phase 3

```bash
python src/crossid_trained.py train-reranker \
  --data-dir crossid_training_data --device cuda \
  --adapter crossid_checkpoints/phase2 \
  --encoder distilbert/distilbert-base-uncased \
  --train-embeddings selma_embeddings/crossid_training/train/train.json \
  --target-embeddings selma_embeddings/crossid_training/train/test_prompt_taskonly.json \
  --dev-train-embeddings selma_embeddings/crossid_training/dev/train.json \
  --dev-target-embeddings selma_embeddings/crossid_training/dev/test_prompt_taskonly.json \
  --output crossid_checkpoints/phase3
```

Use `--revision <model-commit>` to pin the Hugging Face model download. A local
model directory also works. The saved encoder and tokenizer are restored
locally at inference, avoiding a new model download. Training must use the
adapter's exact train/dev embedding files and cohorts.

Defaults: shortlist 32 authors, two nearest references per candidate bundle,
three retrieved negative authors per training target, maximum sequence length
512, batch size 8, and 3 epochs. Positive training bundles prefer references
of the opposite genre when available. The cross-encoder takes the target text
and candidate reference texts as a tokenized pair, without author names or IDs.
Development accuracy counts shortlist misses as failures; it never inserts the
true author into a development shortlist. Best-epoch selection uses this
end-to-end top-1. Pair texts are assembled by batch to bound host memory.

## 5. Fit fusion on calibration authors

```bash
python src/crossid_trained.py calibrate \
  --data-dir crossid_training_data --device cuda --reference-set Both \
  --adapter crossid_checkpoints/phase2 --reranker crossid_checkpoints/phase3 \
  --train-embeddings selma_embeddings/crossid_training/calibration/train.json \
  --target-embeddings selma_embeddings/crossid_training/calibration/test_prompt_taskonly.json \
  --output crossid_checkpoints/fusion_both
```

Fusion combines retrieval score, cross-encoder logit, stylometric cosine
similarity, and negative stylometric distance. Style features cover length,
word lengths/diversity, case, digits, whitespace, and punctuation rates.
Their scaler is fitted on training references only. Candidate feature scaling
and a fixed `C=1` balanced logistic classifier are fitted on calibration authors.
The resulting logit is a ranking score; it is not claimed to be a calibrated
closed-world author probability. Genre and topic leakage can remain despite
the objectives and these features; measure it rather than assuming invariance.

Fusion is specific to the reference condition. Repeat calibration in separate
output directories for Article and Tweet if you want those evaluations.

## 6. Evaluate frozen silver test and exploratory gold

```bash
python src/crossid_trained.py evaluate \
  --data-dir crossid_training_data --device cuda --reference-set Both \
  --adapter crossid_checkpoints/phase2 --reranker crossid_checkpoints/phase3 \
  --fusion crossid_checkpoints/fusion_both/fusion.json \
  --train-embeddings selma_embeddings/crossid_training/test/train.json \
  --target-embeddings selma_embeddings/crossid_training/test/test_prompt_taskonly.json \
  --output results/crossid_trained_silver
```

Evaluate Phase 2 alone by omitting both `--reranker` and `--fusion`. The evaluator
reports SELMA, frozen reference-only, adapted reference-only, and adapted
prototype mixtures, plus cross-encoder-only and fused results when Phase 3 is
provided. Each has Overall/Article/Tweet accuracy, MRR and Recall@8/16/32/64,
compact paired prediction files, and an input/checkpoint/source manifest.
Checkpoint hashes are checked before inference and remain unchanged afterward.
Tie ordering follows sorted author labels. Phase 3 also reports shortlist recall.
The reranker changes only the order of the retrieved authors; all remaining
authors retain their retrieval order and cannot be promoted into the shortlist.

For gold, generate the original splits with `python src/dataset_creation.py`
if they are not already present, and reuse the original gold embedding files:

```bash
python src/crossid_trained.py evaluate \
  --device cuda --reference-set Both \
  --gold-query attribution_data/query/CrossNews_Both.csv \
  --gold-target attribution_data/test/CrossNews.csv \
  --allow-text-overlap \
  --adapter crossid_checkpoints/phase2 --reranker crossid_checkpoints/phase3 \
  --fusion crossid_checkpoints/fusion_both/fusion.json \
  --train-embeddings selma_embeddings/mistral/train.json \
  --target-embeddings selma_embeddings/mistral/test_prompt_taskonly.json \
  --output results/crossid_trained_gold
```

The original gold references and targets share a small number of texts even
though IDs are disjoint. `--allow-text-overlap` explicitly preserves the original
protocol and records overlapping text hashes/counts. ID overlap is always an
error; this option is forbidden for the clean silver test. Gold has already
informed model design through Phase 1 experiments, so these gold results remain
exploratory. The silver test has 60 candidate authors versus gold's 500; raw
accuracy across those datasets is not a directly comparable benchmark.

## Tests and remaining research

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

Install PyTorch and transformers to exercise all training and local Hugging
Face integration tests; otherwise those optional tests are marked skipped.
The tiny transformer is trained and fusion fitted during the CPU fixture test.
No GPU run, production pretrained training, or real Phase 2/3 benchmark has
been completed locally. Full backbone fine-tuning, annotated topic supervision,
large-scale topic-aware negative mining, and multi-seed stability studies remain
future work. These adapter defaults are initial hypotheses, not selected gains.
