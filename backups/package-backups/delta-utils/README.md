# Delta-utils

## Installation

```
pip install delta-utils
```

## To deploy to PyPI

First increase the version number in pyproject.toml (e.g. 0.0.13 -> 0.0.14)

```bash

python3 -m pip install --upgrade build
python3 -m build
python3 -m pip install --upgrade twine
twine upload --skip-existing dist/*

```
