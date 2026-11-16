# candleprep

CPU-only synthetic candle corruption and verified repair-instruction datasets.
The core uses Python's standard library and imports without Torch, a GPU, or a
model. Keep this folder next to `proj5.ipynb` for a normal local import.

```python
from candleprep.datasets import generate_dataset

generate_dataset("TVC_GOLD, 360.csv", "generated_data/v1")
```

Equivalent command, using the intended Python 3.11.9 interpreter:

```powershell
python -m candleprep "TVC_GOLD, 360.csv" --output generated_data/v1
python -m unittest discover -s tests -v
```

For a preview, add `--train 120 --validation 80 --test 100` and select another
output directory. Existing dataset files are never overwritten. CPU generation
needs no pip installs. The notebook uses the existing starter ML dependencies.

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

The notebook implements four independent metrics: recoverability accuracy,
ordered-operation accuracy on oracle-positive examples, strict full-answer
accuracy, and format validity. Routine scoring never executes tools. A separate
explicit final section executes model-predicted plans and reports reconstruction,
false omissions, negative repair attempts, execution failures, and retained coverage.

## Project scope

`proj5.ipynb` retains the starter's Transformers/PEFT Mistral-7B framework and
contains training, metrics, and presentation. GPU stages default to disabled.
Run the tiny smoke/reload check before full training. The planned rank set is
8/16/32, with validation-based selection and a final before/after test comparison.
No model improvement or TAMU compatibility is claimed until actually measured.

The repository includes this library, its tests, the notebook, and the original
CSV. Generated datasets, model artifacts, environments, and local logs are
excluded by `.gitignore`. Generate data after cloning, or copy the generated
artifacts separately when moving execution to TAMU.
