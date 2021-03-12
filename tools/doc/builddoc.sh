# This script must be executed from this folder.
pip install -r ../../docs/python/requirements.txt

# Delete old docs
rm -rf ../../build/docs

# Inference doc
python -m sphinx -j2 -v -T -b html -d ../../build/docs/inference/_doctrees/html ../../docs/python/inference/ ../../build/docs/inference/html
python -u rename_folders.py ../../build/docs/inference/html

# Training doc
python -m sphinx -j2 -v -T -b html -d ../../build/docs/training/_doctrees/html ../../docs/python/training ../../build/docs/training/html
python -u rename_folders.py ../../build/docs/training/html