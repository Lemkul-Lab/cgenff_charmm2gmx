# charmm2gmx_cgenff
Python scripts to convert CGenFF stream files to GROMACS format

# Compatibility
These scripts are compatible with Python2 and Python3, via different versions
that are distributed. Please choose a suitable script based on the version of
Python that you are using.

## Dependencies
The following components are necessary:
* NumPy
* NetworkX (version 1.11 or 2.3)

## Compatible Python versions
The "py2" script is compatible with any Python version in the 2.x series. The
"py3" scripts require a version no newer than 3.7. We have explicitly tested
these scripts with 3.5.x and 3.7.x versions. Newer versions of Python, like 
3.9 and 3.10, will trigger errors related to changes in syntax. We are in the
process of modernizing these scripts, but for now, please do not attempt to
use any Python version newer than 3.7 as a major version.
