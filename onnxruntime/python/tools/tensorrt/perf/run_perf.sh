#!/bin/bash

if [ "$1" == "onnx-zoo-models" ]
then 
    sudo docker run --gpus all -v /home/hcsuser/repos/onnxruntime/onnxruntime/python/tools/tensorrt/perf/:/usr/share/perf/ onnxruntime-trt-perf /bin/bash /usr/share/perf/perf.sh -d /usr/share/perf/ -o onnx-zoo-models
fi 

if [ "$1" == "many-models" ]
then 
    sudo docker run --gpus all -v /home/hcsuser/repos/onnxruntime/onnxruntime/python/tools/tensorrt/perf/:/usr/share/perf/ -v /home/hcsuser/mount/models:/usr/share/mount/many-models onnxruntime-trt-perf /bin/bash /usr/share/perf/perf.sh -d /usr/share/perf/ -o many-models -m /usr/share/mount/models
fi 

if [ "$1" == "partner-models" ]
then 
    sudo docker run --gpus all -v /home/hcsuser/repos/onnxruntime/onnxruntime/python/tools/tensorrt/perf/:/usr/share/perf/ onnxruntime-trt-perf /bin/bash /usr/share/perf/perf.sh -d /usr/share/perf/ -o partner-models
fi
