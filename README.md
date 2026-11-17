# Candle-data repair instructions with PEFT-LoRA

This project prepares synthetic candle-data corruptions and fine-tunes Mistral-7B
to output a JSON tool-use plan:

```json
{"recoverable": true, "operations": ["repair_number_spelling", "parse_number_words"]}
```

Negative examples use `{"recoverable": false, "operations": []}` and prescribe
omitting the entire window. They remain in supervised training.

## Run the notebook

Open `proj5.ipynb` in a CUDA-enabled PyTorch environment with Transformers. The
verified environment used Python 3.11.9, PyTorch 2.10.0+cu128, Transformers 5.8.1,
PEFT 0.21.3, and bitsandbytes 0.50.2. Preserve a working CUDA installation.

The notebook starts by installing the published `candleprep[notebook]==0.2.0`
plus PEFT 0.21.3, Accelerate 1.15.0, and bitsandbytes 0.50.2. Restart the kernel
once after that first installation, then run from the top. These separate
training dependencies are not supplied by candleprep's CPU/display extra.

To repair an existing TAMU notebook session, run:

```python
%pip install "peft==0.21.3" "accelerate==1.15.0" "bitsandbytes==0.50.2"
```

Restart the kernel afterward. The preparation cells then use just:

```python
from candleprep import CandleDataset

dataset = CandleDataset.prepare(output_dir="generated_data/v1", seed=4820)
dataset.display_samples(rows=5)
dataset.display_labels(rows=5)
```

No manual CSV download is needed. The package downloads the original CSV from a
pinned commit of this public repository, verifies its SHA-256, and caches it
inside the dataset directory. Existing matching datasets are verified and reused.
The root CSV remains in Git as the public download source, but the notebook does
not read the working directory's copy. See [the package README](candleprep/README.md).

## Notebook flow

The notebook has 12 code cells / about 300 code lines: package installation;
preparation and data/label previews; settings and cleanup; tokenization/model
helpers; five base-model examples; visible starter-style LoRA training; primary
training; the same five examples afterward; immediate saved-history charts; and
an optional small generated-answer comparison.

Setup disables MathJax for styled data tables so dollar signs in task text and
synthetic currency values display literally. For an already open notebook, run
`import pandas as pd; pd.set_option("styler.html.mathjax", False)` and rerun the
display cell. Cached baseline responses are reused; no retraining is required.

The SFT dataset contains 10,000 corrupted-input/verified-JSON-label pairs.
Deterministic reconstruction proves positive labels offline. Originals/history
never enter model prompts. Target-only loss masking preserves full answers.
Trainer consumes cached labels without generating responses during training.

Rank-16 primary training uses batch size 1, accumulation 16, one epoch / 625
optimizer steps, gradient checkpointing, and checkpoints every 100 steps.
Completed adapters are reused and interrupted training resumes from the latest
checkpoint. New simplified runs use `candle_results/simple_sft_v2/`; earlier
`full_sft_v1/` adapters/results remain untouched. Settings changes require a new
RESULTS directory.

Configuration settings use one variable per line. `LORA_RANK` chooses the
single adapter trained by this run. There is no automatic rank sweep. For the
assignment's rank experiment, change `LORA_RANK` and rerun Sections 1–9; ranks
are saved separately. Use a new `RESULTS` directory for changed training
hyperparameters such as learning rate or epoch count, and document observations.
The handout requires rank exploration but does not prescribe 8/16/32 or a sweep.

Five validation cases are fixed before generation and reused afterward. Their
before/after metrics are saved, with trained results stored per rank. Section 10
reads saved Trainer history and metric files to plot training/validation loss,
teacher-forced target-token accuracy, and the five-case generated-answer results.
No model, inference job, or optional comparison is needed to draw those charts.

New runs use `VALIDATION_EXAMPLES=64` fixed validation cases, every
`VALIDATION_EVERY=100` optimizer steps and at training completion. Trainer makes
teacher-forced forward passes against cached labels; it does not generate answers
or update weights during validation. Only argmax token IDs are retained for
metrics, with evaluation batch size 1. Target-token accuracy excludes prompt and
padding tokens and accounts for the causal next-token shift. It is a different
metric from complete generated-JSON correctness.

Older completed adapters are reused. Their training curves remain displayable;
if they did not record validation loss/accuracy, those missing metrics are marked
as unavailable. The notebook does not fabricate a validation trajectory or rerun
training to create one. Rerunning Section 6 reuses frozen responses and writes
its small baseline metric file; Section 9 saves the five trained-case metrics.

Section 11 is optional: `COMPARISON_EXAMPLES=0` skips extra generation; set it to
25 for 25 deterministic held-out cases per model. The original base and trained
adapter receive the same sample. Results record case IDs, count, predictions,
and metrics under the selected rank. This does not claim evaluation of all
2,000 test examples. The prior full-dataset generation and dependent final-tool
execution gate have been removed from the notebook; candleprep's explicit
repair-execution API remains available separately.

Standard Trainer progress reports steps/loss; inference reports examples/ETA.
After an interruption has stopped, rerun the Manual GPU cleanup cell. It defines
and calls `free_gpu_resources()` immediately; redefining the function alone does
not run cleanup. You can also call `free_gpu_resources()` in a new cell. Cleanup
also clears bitsandbytes' global optimizer registrations, which can retain model
layers after an interrupted training run, then collects remaining reference cycles
and releases cuBLAS workspace caches when supported by PyTorch. The
same cache-release function runs automatically after each model phase. The
printed allocation/reservation describes the current PyTorch kernel; CUDA context
and other applications may still use memory reported by Task Manager.
After a kernel restart, rerun setup through Section 5, Section 6 (it reuses saved
baseline responses), and the Section 7 training-function definition. Section 9
can then reload the saved adapter without retraining.

The prior rank-16 run recorded 5.90 GiB peak PyTorch allocation and the live
kernel showed 6.17 GiB peak reservation afterward. Whole-device peak was not
logged; the other 10-GB machine still requires verification. The simplified
loader reports available memory and uses the same low-memory configuration,
without an arbitrary minimum-capacity rejection. Training saves allocated and
reserved peaks. No full training has been rerun for this notebook refactor.

## Verification and package release

The test suite covers data preparation, training feedback, and the lightweight workflow.
A clean working directory with no root CSV downloaded and generated all 13,000
examples successfully. Artifact hashes match the earlier dataset exactly.
Package tests cover deterministic reconstruction, split isolation, checksum/cache
validation, public-feature separation, scoring, displays, and mocked training.
The cached tokenizer check covers all 10,000 training examples (longest: 1,019
tokens within the 1,536 limit). Run `python -m unittest discover -s tests -v`.

Release [0.2.0 is published on PyPI](https://pypi.org/project/candleprep/0.2.0/)
and matches the checked local archives. Version 0.1.0 has been yanked; use 0.2.0.
See [PUBLISHING.md](PUBLISHING.md) for building, checking, and uploading.

Instructor starter code/PDF, datasets, model weights/checkpoints, environments,
temporary files, and local logs remain excluded from Git. Package distributions
contain no CSV, notebook, instructor materials, or generated/model artifacts.
The CPU library is MIT licensed.
