# candleprep

CPU-only synthetic candle corruption and verified repair-instruction datasets.
Python 3.11+, MIT licensed. The core uses the standard library and imports
without Torch, Transformers, pandas, a model, or a GPU.

## Installation

Published release 0.2.0 adds `CandleDataset` and automatic source downloading.
Version 0.1.0 has been yanked; install 0.2.0:

```bash
python -m pip install "candleprep[notebook]==0.2.0"
```

The optional `notebook` extra supplies pandas, matplotlib, IPython, datasets, and
tqdm for display/conversion. It does not install a CUDA/PyTorch training stack.
For CPU preparation alone, install `candleprep==0.2.0` without extras.
A source checkout can be installed with `python -m pip install .`.

## Prepare and inspect in a few calls

```python
from candleprep import CandleDataset

dataset = CandleDataset.prepare(output_dir="generated_data/v1", seed=4820)
dataset.display_samples(rows=5)
dataset.display_labels(rows=5)
```

No local source CSV is required. `prepare()` downloads the original
[`TVC_GOLD, 360.csv`](https://github.com/Absolute-Convergence/NLP-proj5/blob/94b97c68550b11aa6eeb1b0d948cc6cda1ac1a3a/TVC_GOLD%2C%20360.csv)
from a pinned NLP-proj5 commit. Its SHA-256 is
`aa89b7a05abad8f3ee718c65607169a547ec215370cde15cfe98f8c3a895633c`.
The CSV is cached as `output_dir/source.csv`. Matching generated data is reused
after source/settings/artifact verification; mismatches raise an error rather
than silently overwriting a run. First use requires internet; verified caches
can be reused offline.

Defaults are 10,000 train / 1,000 validation / 2,000 test examples and four source
rows per window. Override `train_size`, `validation_size`, `test_size`, and
`window_size` for a different dataset, using a different `output_dir`.
An explicit `source="path/to/custom.csv"` still supports other local inputs.

`display_samples()` shows first/middle/last groups for both original and corrupt
data. `display_labels()` executes positive labels and displays reconstruction
against the oracle's original, alongside whole-window omission decisions.
Display methods import their optional dependencies only when called.

## Use the records for training and evaluation

```python
training_records = dataset.records("train")  # only id, input, target
examples = dataset.examples(count=5)         # fixed validation cases
# predictions = ... model-generated JSON strings for dataset.records("test")
# report = dataset.evaluate(predictions, split="test")
```

`candleprep.task.user_text(record)` formats the static repair task plus corrupted
CSV only. JSON scoring lives in `candleprep.evaluation`; it does not execute tools.
`dataset.evaluate()` also returns category/composition breakdowns. The separately
called `dataset.execute_predictions(predictions, destination)` is the final tool
execution/reconstruction demonstration. Generation, LoRA, GPU management, and
Trainer remain in the notebook.

The lower-level 0.1.0 API and CLI remain available:

```python
from candleprep.datasets import generate_dataset, load_jsonl
generate_dataset("path/to/source.csv", "generated_data/custom")
```

```bash
python -m candleprep path/to/source.csv --output generated_data/custom
python -m candleprep --version
```

## Data and output contract

Each input is raw CSV with a header and four consecutive original records.
`time`, `open`, `high`, `low`, `close`, and `Volume` are required. `ROC` is optional:
the source has 118 existing ROC blanks in 20,330 records. The original file is
never cleaned in place or modified. Baseline audit found unique, increasing
timestamps and no basic OHLC/volume violations. UTC offsets are explicit;
normal irregular market intervals are allowed.

Labels contain exactly two fields:

```json
{"recoverable": true, "operations": ["repair_number_spelling", "parse_number_words"]}
```

```json
{"recoverable": false, "operations": []}
```

Negative examples stay in supervised data. A negative decision means omit the
complete window during cleaning, including all its rows and fields.

`train.jsonl`, `validation.jsonl`, and `test.jsonl` contain only `id`, `input`, and
`target`. IDs and targets are not prompt features. The corresponding
`*.references.jsonl` files contain privileged originals, source row identities,
history, and evaluation metadata. Never feed those sidecars to the model.
`manifest.json` records the source hash, seed, exact counts, recipes, exclusions,
operation coverage, source boundaries, and hashes of the exported files.

Split source records chronologically 70/10/20 before constructing non-overlapping
windows. Every derivative of a source window stays in its partition. A fixed
seed reproduces the same artifacts. Duplicate model-visible inputs and conflicting
labels are rejected and counted. Lexical recipes receive substantial coverage;
the manifest reports actual class proportions rather than assuming balance.

## Supported behavior

| Category | Supported examples | Information loss / unsupported recovery |
|---|---|---|
| Syntactic | Remove declared `? @ ~` noise; repair doubled decimal points | Arbitrary digits replaced, deleted, or transposed; lost decimal |
| Formatting | `$`, USD/contracts/percent suffixes, decimal comma, explicit-offset alternate dates, whitespace, scientific notation | Thousands grouping and ambiguous locale conventions |
| Structural | Columns reordered with headers; cyclic row rotations with one identifiable timestamp; consistent comma/semicolon/pipe dialect | Unknown cell swaps, lost/split/merged field boundaries |
| Semantic | Explicit cents-to-base-unit scaling | Unmarked sign/scale/outlier/context changes |
| Completeness | Normalize optional missing-value representations | Missing required values, truncated values, lost columns |
| Integrity | Exact duplicate records by timestamp; chronological sorting | Conflicting duplicates, overwritten timestamps, lost rows |
| Lexical | English cardinal/sign/fractional forms, currency words, mixed digits/words, conjunctions, hyphens, case, spacing, controlled spelling variants | Ambiguous/unsupported spellings, destroyed fractional content |

English integers have magnitude below 10^15. Fractional parts use individually
spoken digits, preserving exact decimal meaning. Examples include `hundrad`,
`hundrd`, `fify-three`, `fourty`, and unique one-edit substitutions, insertions,
deletions, transpositions, or repetitions. Spelling correction and parsing remain
separate operations. The decimal-comma rule never assumes thousands separators.
Alternate dates use `DD/MM/YYYY HH:MM:SS%z`; ISO timestamps preserve UTC instants.

All public recovery functions accept and return the same CSV text payload.
Validator functions return unchanged text on success and raise `RecoveryError`
on failure. Detection alone never creates an original value. `detect_outliers`
currently checks domain constraints, not statistical anomaly detection.

## Oracle and evaluation

The oracle deliberately sees originals and full corruption history. Public
recovery tools receive only corrupted text and the fixed schema. The planner
checks short recipe subsequences in reverse-history and declared dependency
order, shortest first, with lexical operation-name tie-breaking. Every positive
plan reconstructs the complete reference under exact Decimal values, UTC
instants, row order, and original optional missingness. Unsupported recovery is
distinct from programming errors, which propagate and stop generation.

This is a bounded supported-tool policy, not proof that no possible repair
algorithm exists. Provenance marks information-destroying cases for omission.
The model cannot necessarily identify a plausible numeric error from the input
alone; that uncertainty must be analyzed in the experiment.

The CPU evaluation module implements four independent metrics: recoverability accuracy,
ordered-operation accuracy on oracle-positive examples, strict full-answer
accuracy, and format validity. Routine scoring never executes tools. The explicit
`dataset.execute_predictions()` API executes model-predicted plans and reports reconstruction,
false omissions, negative repair attempts, execution failures, and retained coverage.

## Project scope

`proj5.ipynb` retains the starter's Transformers/PEFT Mistral-7B framework and
contains visible training and experiment presentation. There is no small-training
prerequisite: it trains the configured primary rank directly from offline labels.
Five fixed validation cases are compared before and after SFT. Rank experiments are performed manually by changing the rank and rerunning the
notebook. Training/validation charts read saved Trainer history directly. A small
held-out generation sample is optional; no full-dataset inference job gates results.
No model improvement or TAMU compatibility is claimed until actually measured.

The repository includes this library, its tests, the notebook, and the original
CSV. Generated datasets, model artifacts, environments, and local logs are
excluded by `.gitignore`. The notebook can download the original and generate
its dataset automatically, including when run without a repository clone.
