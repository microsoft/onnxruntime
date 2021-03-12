@echo off
rem This script must be executed from this folder.
pip install -r ../../docs/python/requirements.txt
python -m sphinx -j2 -v -T -b html -d ../../build/ortmodule_docs/_doctrees/html ../../docs/python/ortmodule ../../build/ortmodule_docs/html
python -u rename_folders.py ../../build/ortmodule_docs/html
