# This script must be executed from this folder.
# $1 python path
# $2 source folder
# $3 build folder
echo "----"
echo $1
echo $2
echo $3
echo "----"
ls $3/Release
echo "----"
$1/python -m pip install -r $2/docs/python/requirements.txt
export PYTHONPATH=$3/Release:$PYTHONPATH
$1/python -m sphinx -j1 -v -T -b html -d $3/docs/_doctrees/html $2/docs/python $3/docs/html
$1/python -u $2/tools/doc/rename_folders.py $3/docs/html
# zip -r $3/python_doc.zip $3/docs/html