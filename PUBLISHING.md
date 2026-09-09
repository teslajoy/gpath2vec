# Publishing gpath2vec to PyPI

As of 2026-06-17 `gpath2vec` is **not yet on PyPI** (the JSON API returns 404),
so this is a first-time publish and the name appears free. Versions on PyPI are
immutable: you can never re-upload the same version, only bump and upload again.

## 0. one-time setup

- Create accounts on https://pypi.org and (for a dry run) https://test.pypi.org.
- Create an **API token** on each: Account settings -> API tokens -> "Add API
  token" (scope it to this project after the first upload). Use tokens, not your
  password.
- Optionally store them in `~/.pypirc`:

  ```ini
  [pypi]
    username = __token__
    password = pypi-AgEI...your-token...

  [testpypi]
    username = __token__
    password = pypi-AgEN...your-test-token...
  ```

- Install the build tooling (neither is currently installed in this venv):

  ```bash
  pip install --upgrade build twine
  ```

## 1. bump the version

Edit `setup.py` -> `__version__`. Use PEP 440 (e.g. `3.0.0` for the first
release, `3.0.1`, `3.1.0`, ...). The first PyPI release can be any version.

## 2. build the distributions

```bash
rm -rf dist build *.egg-info        # clean stale artifacts first
python -m build                     # writes dist/gpath2vec-X.Y.Z.tar.gz + .whl
twine check dist/*                  # validates metadata + README rendering
```

`setup.py` uses `find_packages()`, so only the `gpath2vec/` package is shipped.
`scripts/`, `tests/`, `img/`, and the paper drafts are NOT included (and
`scripts/` is gitignored). The `gpath2vec` console command is installed from
`entry_points`. The long description is `README.md`, rendered on the project
page, so keep it clean.

## 3. dry run on TestPyPI (recommended)

```bash
twine upload --repository testpypi dist/*
# in a fresh venv, confirm it installs and the CLI works:
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple gpath2vec
gpath2vec --help
```

(The `--extra-index-url` lets dependencies resolve from real PyPI while the
package itself comes from TestPyPI.)

## 4. publish to PyPI

```bash
twine upload dist/*
```

## 5. verify the install

```bash
pip install gpath2vec              # core (Fisher path)
pip install 'gpath2vec[aucell]'    # adds decoupler + anndata (AUCell path)
gpath2vec --help
```

## install from source (no PyPI)

```bash
pip install -e .                   # core
pip install -e '.[aucell]'         # with AUCell extra
```

## notes / optional improvements

- Consider adding a minimal `pyproject.toml` with a `[build-system]` table for a
  fully modern build; `setup.py` still works with `python -m build`.
- To ship demo scripts with the package, move them under `gpath2vec/` or add a
  `MANIFEST.in` / `package_data`; right now they are excluded by design.
- Tag the release in git to match the PyPI version (e.g. `git tag v3.0.0`).
- For an archival DOI (Zenodo), connect the GitHub repo to Zenodo and cut a
  GitHub release; cite that DOI in the paper's data-availability section.
