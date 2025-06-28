# cgenff_charmm2gmx

Python scripts to convert CGenFF stream files to GROMACS format

## Compatibility

This script was updated for Python 3.13+. It should also work for 3.11 and onwards. A newer Python version is required as the script has been updated to include complete type hinting.

For older versions of Python see the legacy_scripts folder.

## Dependencies

The following components are necessary:

* NumPy
* NetworkX

This can be easily setup using pip (we recommend using a virtual environment) or with [uv](https://docs.astral.sh/uv/).

#### pip

    pip install -r requirements.txt

#### uv

    uv sync # To use the exact versions this script was tested with

OR

    uv venv # To create a virtual environment
    uv pip install -r pyproject.toml # Use the latest versions of the dependencies

### Running the script

Usage: RESNAME drug.mol2 drug.str charmm36.ff

Example:

    python cgenff_charmm2gmx.py JZ4 jz4.mol2 jz4.str charmm36_ljpme-jul2022.ff

OR

    uv run cgenff_charmm2gmx.py JZ4 jz4.mol2 jz4.str charmm36_ljpme-jul2022.ff

### Run testing

Pytest will run the script and compare the results to the output from the old Python 3.7 output.

    pytest

OR to get coverage info

    pytest --cov --cov-report term:skip-covered --cov-report html
