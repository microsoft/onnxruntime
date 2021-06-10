echo "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES" 
echo "PythonManylinuxDir=$PythonManylinuxDir"

ls -l /dev/kfd
ls -l /dev/dri

$PythonManylinuxDir/bin/python3 -m pip install /build/Release/dist/*.whl 
/onnxruntime_src/tools/doc/builddoc.sh $PythonManylinuxDir/bin/ /onnxruntime_src /build Release  
