# Publishing candleprep 0.2.0

Version 0.2.0 adds the CandleDataset facade, pinned/checksummed source download,
optional notebook displays, and CPU instruction scoring. The existing generator
and CLI remain available. Core preparation still has no runtime dependencies;
the optional notebook extra supplies display/dataset utilities.

Release [0.2.0 is published](https://pypi.org/project/candleprep/0.2.0/), and the
published wheel/source archive checksums match the checked local builds. Version
0.1.0 is yanked. The notebook's pinned `candleprep[notebook]==0.2.0` cell is ready.
The package version comes from `candleprep/__init__.py`.

The build/upload instructions below are retained for future releases. Do not
re-upload 0.2.0; increment the version before the next package release.

## Verified locally

All 44 tests pass. Both 0.2.0 archives pass `twine check --strict` and the
distribution-content audit. The wheel installs into a clean Python 3.11
environment, imports without ML/display dependencies, and downloads/prepares
data outside the source checkout. A separate full 13,000-example generation in
a working directory without the root CSV reproduced all earlier dataset artifact
hashes. The underlying corruption/oracle algorithms and task prompt are unchanged.
Full GPU training has not been rerun for this refactor.

## Build and check

Use the separate tooling environment so the working CUDA environment is unchanged.
From the repository root in PowerShell:

```powershell
.\.packaging-venv\Scripts\python -m build
.\.packaging-venv\Scripts\python -m twine check --strict dist/candleprep-0.2.0.tar.gz dist/candleprep-0.2.0-py3-none-any.whl
python tools/check_distribution.py
```

If that environment is missing, create it with `python -m venv .packaging-venv`
and install build tools with `.\.packaging-venv\Scripts\python -m pip install build twine`.
On Linux/macOS, use `.packaging-venv/bin/python` instead.

The checker/uploader select only the version in the current source, so retained
0.1.0 artifacts are not accidentally uploaded. Archives contain the package,
package README, license, and build metadata; no dataset, CSV, notebook, model,
instructor materials, tests, or local credentials are included.

## Upload

```powershell
.\.packaging-venv\Scripts\python tools/upload_package.py
```

Enter a production PyPI token authorized for `candleprep` at the hidden local
prompt. The helper checks the two current-version archives and passes the entered
token to Twine in memory, overriding keyring/config credentials. It does not
write the token to source or configuration. An empty token cancels the upload.
The earlier 403 project-scope error means a token was not authorized for this
project; use a candleprep-scoped token or an authorized account-wide token.
Never paste a token into the notebook or commit it to Git.

After successful upload, verify installation from outside a source checkout:

```powershell
python -m pip install "candleprep[notebook]==0.2.0"
python -c "from candleprep import CandleDataset; print(CandleDataset.prepare().summary())"
```

This downloads and verifies the original source CSV automatically. First use
requires internet access; cache reuse does not. This step does not train a model.
