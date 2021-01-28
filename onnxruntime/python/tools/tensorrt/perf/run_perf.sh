#!/bin/bash

if [ "$1" == "onnx-zoo-models" ]
then 
    sudo docker run --gpus all -v /home/hcsuser/repos/onnxruntime/onnxruntime/python/tools/tensorrt/perf/:/usr/share/perf/ 36f6f9268db2 /bin/bash /usr/share/perf/perf.sh -d /usr/share/perf/ -o onnx-zoo-models
fi 

if [ "$1" == "many-models" ]
then 
    sudo docker run --gpus all -v /home/hcsuser/repos/onnxruntime/onnxruntime/python/tools/tensorrt/perf/:/usr/share/perf/ -v /home/hcsuser/mount/many-models:/usr/share/mount/many-models 36f6f9268db2 /bin/bash /usr/share/perf/perf.sh -d /usr/share/perf/ -o many-models -m /usr/share/mount/many-models
fi 

if [ "$1" == "partner-models" ]
then 
    sudo docker run --gpus all -v /home/hcsuser/repos/onnxruntime/onnxruntime/python/tools/tensorrt/perf/:/usr/share/perf/ 36f6f9268db2 /bin/bash /usr/share/perf/perf.sh -d /usr/share/perf/ -o partner-models
fi
