# Publishing candleprep

The first prepared release is `candleprep` version **0.1.0**, licensed under MIT.
It requires Python 3.11 or newer and has no runtime dependencies. The version
comes from `candleprep/__init__.py`.

## Prepared release verification

Both archives pass `twine check --strict` and the distribution-content audit.
The wheel installed into a clean Python 3.11.9 environment; the source archive
built and installed into a clean Python 3.13.14 environment. In both environments,
all 26 package-relevant tests passed, the installed API reproduced the complete
13,000-example dataset and its artifact hashes, and the installed CLI generated
a 300-example preview. Imports resolved to the installed package, with no Torch
import or runtime dependency installation. These checks were on Windows;
execution on TAMU's Linux environment remains to be verified.

| File | Bytes | SHA-256 |
|---|---:|---|
| `candleprep-0.1.0-py3-none-any.whl` | 30,105 | `2c7d66fea3bebd08c9ca94efa835b3a45d65fbd641f33ac6db9dc40617cb63ef` |
| `candleprep-0.1.0.tar.gz` | 24,896 | `093722f08671a70de0df7a8b4998f301b8d3cf62ec40081e71832de181f729a8` |

Release 0.1.0 was published by the user and is available at
<https://pypi.org/project/candleprep/0.1.0/>. The hashes above describe the initial
local build; rebuilding can change archive hashes. Published files have their
own checksums on PyPI. Do not re-upload an already published version's filenames.

## Build and check

Use a separate tooling environment if build tools are not already installed.
This avoids changing the notebook's working CUDA environment.

Windows PowerShell:

```powershell
python -m venv .packaging-venv
.\.packaging-venv\Scripts\python -m pip install build twine
.\.packaging-venv\Scripts\python -m build
.\.packaging-venv\Scripts\python -m twine check --strict dist/candleprep-0.1.0.tar.gz dist/candleprep-0.1.0-py3-none-any.whl
python tools/check_distribution.py
```

Linux/macOS:

```bash
python3 -m venv .packaging-venv
.packaging-venv/bin/python -m pip install build twine
.packaging-venv/bin/python -m build
.packaging-venv/bin/python -m twine check --strict dist/candleprep-0.1.0.tar.gz dist/candleprep-0.1.0-py3-none-any.whl
python3 tools/check_distribution.py
```

The wheel and source archive contain only package code, package documentation,
licensing, and build metadata. They exclude the notebook, CSV, generated data,
tests, instructor materials, environments, model artifacts, and local logs.
`dist/` and the tooling environment stay out of Git.

## Upload when ready

For the first upload of this new project, create a production PyPI token with
scope **Entire account (all projects)** at <https://pypi.org/manage/account/token/>.
A token scoped to another project cannot create `candleprep`. Once the project
exists, use a token scoped specifically to `candleprep` for later uploads.

Run the helper below from the repository root. It validates both artifacts and
always prompts for a token, overriding cached keyring and `.pypirc` credentials.
Enter the token at the hidden prompt. It is passed to Twine in memory, without
writing it to source, configuration, or the shell command/history. Existing saved
credentials are left untouched.

Windows PowerShell:

```powershell
.\.packaging-venv\Scripts\python tools/upload_package.py
```

Linux/macOS:

```bash
.packaging-venv/bin/python tools/upload_package.py
```

These commands upload to production PyPI. Building and checking do not publish
anything. A successful first upload creates the PyPI project; the name is not
reserved before that. If a version has already been published, increment
`__version__`, rebuild, and upload the new version's exact files.

If an earlier direct `twine upload` reports `password set from keyring`, it reused
a saved token instead of prompting. An error saying `project-scoped token is not
valid for project: 'candleprep'` identifies the wrong token scope. Use the helper
with a correctly scoped token; the package does not need rebuilding for this error.

## Install elsewhere after publication

```bash
python -m pip install candleprep==0.1.0
candleprep --version
```

In a notebook, use `%pip install candleprep==0.1.0` once, then restart the kernel
if required. The package does not install a model, notebook, or source CSV. Get
the notebook and CSV from the GitHub repository, or provide your own matching
CSV. Dataset generation then uses the same API everywhere:

```python
from candleprep.datasets import generate_dataset

generate_dataset("TVC_GOLD, 360.csv", "generated_data/v1")
```

References: [Python packaging guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/),
[pyproject metadata](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/),
[Twine](https://twine.readthedocs.io/en/stable/).
