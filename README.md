# Candle-data repair instructions with PEFT-LoRA

This project creates synthetic candle-data corruption examples and trains a
Mistral-7B model to predict whether a complete example is recoverable and which
registered recovery functions to apply. The output is a JSON instruction object:

```json
{"recoverable": true, "operations": ["repair_number_spelling", "parse_number_words"]}
```

Non-Recoverable examples use `{"recoverable": false, "operations": []}` and
prescribe omission of the complete input window during cleaning. They remain in
the supervised dataset.

## Included files

- `candleprep/`: standalone CPU corruption, recovery, oracle, and dataset library.
- `proj5.ipynb`: dataset inspection, scoring, and guarded Transformers/PEFT experiments.
- `tests/`: corruption, oracle, generation, scoring, and notebook checks.
- `TVC_GOLD, 360.csv`: unchanged source candle data.

The instructor's starter notebook and assignment PDF are excluded. Environments,
generated datasets, model weights/checkpoints, temporary files, and local logs
are also excluded by `.gitignore`.

## Generate the dataset

The CPU library uses Python's standard library. Install a source checkout with
`python -m pip install .`, or install the published package with
`python -m pip install candleprep==0.1.0`. Requires Python 3.11 or newer.
Run from this repository's root:

```powershell
python -m candleprep "TVC_GOLD, 360.csv" --output generated_data/v1
```

This creates 10,000 training, 1,000 validation, and 2,000 test examples, plus
oracle reference sidecars and a reproducibility manifest. Source records are
split chronologically before creating four-record windows and corruption variants.
The original CSV is never modified. Existing generated files are protected from
overwrite; use another output directory to generate a separate dataset.

See [the library README](candleprep/README.md) for supported operators, recovery
rules, output contracts, and limitations.

## Run the notebook and checks

Open `proj5.ipynb` using a Python environment with the existing Transformers,
Datasets, PEFT, PyTorch/CUDA, bitsandbytes, Accelerate, pandas, matplotlib, and
IPython dependencies. The local verification environment used Python 3.11.9,
PyTorch 2.10.0+cu128, Transformers 5.8.1, Datasets 5.2.0, PEFT 0.21.3, and
bitsandbytes 0.50.2. Preserve a working CUDA installation when configuring a new
machine; the CPU generator itself needs none of these ML packages.

The notebook shows the first, middle, and last five original rows and synthetic
examples. Section 4 constructs the SFT dataset and displays deterministic repair
verification alongside its JSON labels. It can generate missing data with one
library call. The later model cells load the untrained base, show three fixed
input/output examples, train on all 10,000 pairs, and show five examples after
SFT: the same three plus two additional repair cases. Run cells in order; the
full GPU training cells can take several hours.
The final model-directed repair demonstration remains separately disabled.

The main training path uses the cached input/JSON-label pairs directly with
Trainer; it does not generate responses to construct training labels. One
rank-16 adapter is trained by default. Section 9 generates three baseline
responses; Section 11 generates five trained-model inspection responses, with
paired metrics kept separate from five-case inspection metrics. Full rank studies and full held-out
response generation are explicitly enabled evaluation jobs in Section 12,
separate from dataset construction and SFT.

After editing `RUN_RANK_STUDY` or `RUN_FULL_TEST_COMPARISON`, rerun the
configuration cell, then execute the corresponding Section 12 cell. Editing
notebook text does not update kernel variables or launch those jobs. Both the
configuration and optional cells print the active values.

The completed rank-16 run recorded a 5.90 GiB PyTorch allocation peak; its live
kernel showed a 6.17 GiB reservation peak after training and inference. Whole-device
peak usage was not logged, so these measurements do not certify a 10-GB device.
The free-memory guard is configurable as `MIN_FREE_VRAM_GIB=7.5`, replacing the
previous 12-GiB requirement. Keep the current batch and sequence settings for a
10-GB trial; other GPUs and ranks still need verification. Future training reports
record both allocated and reserved peaks.

Long inference phases show named example/token progress, elapsed time, rate, and
ETA. Training reports optimizer steps and accumulation activity, with loss on the
first step and every five steps. Progress code applies to new calls after the
updated helper cells are executed; it cannot change an already-running call.

After interrupting a run, wait for the kernel to become idle and run
`free_gpu_resources()` in a new cell. The function is defined near the top of
the notebook. It drops the notebook's live model/training
references, clears cached Python results and exception traceback locals, then
collects garbage and releases unused CUDA cache. It prints remaining allocated
and reserved memory. Dataset variables and saved checkpoints remain available;
rerun the interrupted execution cell to reload its model. If CUDA still reports
an error, restart the kernel.

After a kernel restart, the adapter files remain saved but the setup variables
are gone. Follow the checklist at the start of Section 11: rerun Sections 1–8
and the Section 9 helper-definition cell, then run Section 11. Skip the actual
baseline-generation and SFT execution cells. Also restore the Section 10
training-function definition before starting a rank study. The later execution
cells now explain missing prerequisites instead of failing on the first helper name.

After generating the dataset, run:

```powershell
python -m unittest discover -s tests -v
```

The notebook tokenizer test additionally requires the Mistral tokenizer in the
local Hugging Face cache. Loading the notebook's tokenizer once prepares that
cache. Model experiments require a CUDA GPU and downloaded model weights.

## Verification status

All 41 tests pass. All 13,000 generated inputs are unique, source
rows do not cross dataset splits, and all 8,016 oracle-positive examples
reconstruct their complete references. Regeneration reproduces artifact hashes.

The initial pipeline check is complete and its small training path has been
removed from the notebook. Three real untrained-model validation responses are
saved with exact prompts and targets; all three failed the strict JSON/output
contract. This is a small illustrative baseline, not a full test-set result.
Rank-16 SFT completed all 625 steps on 10,000 examples (one epoch), with training
loss 0.02827. All three matched trained-model outputs satisfied the strict answer
contract; the before/after demonstration remains too small to estimate held-out
accuracy. Full quantitative evaluation, rank comparisons, and TAMU A30
verification remain pending. Model artifacts and checkpoints stay outside Git.
The expanded five-case after-SFT inspection also matched all expected JSON
answers. The suite covers notebook syntax, example selection/display, SFT workflow,
restart prerequisites, interruption cleanup, progress feedback, and the free-memory guard.

## Package release

The standalone library is MIT licensed. See [PUBLISHING.md](PUBLISHING.md) for
building, checking, uploading, and installing the package on another machine.
The PyPI distributions exclude the CSV, notebook, instructor materials, and all
large/generated artifacts.
