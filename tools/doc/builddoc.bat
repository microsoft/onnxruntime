@echo off
rem This script must be executed from this folder.
pip install -r ../../docs/python/requirements.txt
python -m sphinx -j2 -v -T -b html -d ../../build/docs/_doctrees/html ../../docs/python ../../build/docs/html
python -u rename_folders.py ../../build/docs/html
