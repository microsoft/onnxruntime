# This script must be executed from this folder.

# $1 python path
# $2 source folder
# $3 build folder
# $4 build config

# Fail the document generation if anything goes wrong in the process
set -e -x

# Install doc generation tools
$1/python -m pip install -r $2/docs/python/requirements.txt

# Fake onnxruntime installation
export PYTHONPATH=$3/$4:$PYTHONPATH

# Remove old docs
rm -rf $3/docs/

# Inference doc
# $1/python -m sphinx -j1 -v -T -b html -d $3/docs/inference/_doctrees/html $2/docs/python/inference $3/docs/inference/html
# $1/python -u $2/tools/doc/rename_folders.py $3/docs/inference/html
# (cd $3/docs/inference/html && zip -r $3/docs/python_inference_doc.zip .)

# Training doc
$1/python -m sphinx -j1 -v -T -b html -d $3/docs/training/_doctrees/html $2/docs/python/training $3/docs/training/html
$1/python -u $2/tools/doc/rename_folders.py $3/docs/training/html
# (cd $3/docs/training/html && zip -r $3/docs/python_training_doc.zip .)
