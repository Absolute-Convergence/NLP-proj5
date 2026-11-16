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

The CPU library uses Python's standard library. Run from this repository's root:

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

The notebook shows the first, middle, and last 20 original rows and synthetic
examples. It can generate missing dataset artifacts with one library call. All
GPU stages default to disabled; enable only the intended experiment.

After generating the dataset, run:

```powershell
python -m unittest discover -s tests -v
```

The notebook tokenizer test additionally requires the Mistral tokenizer in the
local Hugging Face cache. Loading the notebook's tokenizer once prepares that
cache. Model experiments require a CUDA GPU and downloaded model weights.

## Verification status

The CPU suite has 28 passing tests. All 13,000 generated inputs are unique, source
rows do not cross dataset splits, and all 8,016 oracle-positive examples
reconstruct their complete references. Regeneration reproduces artifact hashes.

A brief rank-8 GPU smoke run completed two optimizer steps, saved and reloaded
the adapter, and generated one validation response. The response was valid JSON
but incorrect. Full training, before/after evaluation, rank comparisons, and TAMU
A30 verification remain pending. The smoke run does not establish improved task
accuracy. Its compact results are visible in the notebook; model artifacts are
kept outside Git.
