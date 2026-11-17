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

The notebook has 13 code cells / 256 code lines (including a separate cleanup
definition cell): package installation; preparation;
data/label previews; model settings and cleanup; tokenization/model helpers; five
base-model examples; visible starter-style LoRA training; primary training;
the same five examples afterward; rank comparison; full test comparison; results.

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

Five validation cases are fixed before generation and reused afterward. Rank
8/16/32 validation studies and base/adapter full-test inference remain separate
long jobs; both switches are currently True. They do not create training labels
or delay primary SFT. Change a switch, rerun settings, then execute its job cell.

Standard Trainer progress reports steps/loss; inference reports examples/ETA.
After an interruption has stopped, run `free_gpu_resources()` in a new cell.
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

All 44 tests pass. A clean working directory with no root CSV downloaded and generated all 13,000
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
