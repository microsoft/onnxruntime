#!/bin/bash

while getopts d:o:m: parameter
do case "${parameter}"
in 
d) DOCKER_IMAGE=${OPTARG};;
o) OPTION=${OPTARG};;
m) MODEL_PATH=${OPTARG};;
esac
done 

if [ $OPTION == "onnx-zoo-models" ]
then 
    sudo docker run --gpus all -v /home/hcsuser/repos/onnxruntime/onnxruntime/python/tools/tensorrt/perf/:/usr/share/perf/ $DOCKER_IMAGE /bin/bash /usr/share/perf/perf.sh -d /usr/share/perf/ -o onnx-zoo-models -m model.json
fi 

if [ $OPTION == "many-models" ]
then 
    sudo docker run --gpus all -v /home/hcsuser/repos/onnxruntime/onnxruntime/python/tools/tensorrt/perf/:/usr/share/perf/ -v /home/hcsuser/mount/models:/usr/share/mount/many-models $DOCKER_IMAGE /bin/bash /usr/share/perf/perf.sh -d /usr/share/perf/ -o many-models -m /usr/share/mount/many-models
fi 

if [ $OPTION == "partner-models" ]
then 
    sudo docker run --gpus all -v /home/hcsuser/repos/onnxruntime/onnxruntime/python/tools/tensorrt/perf/:/usr/share/perf/ $DOCKER_IMAGE /bin/bash /usr/share/perf/perf.sh -d /usr/share/perf/ -o partner-models -m partner_model_list.json
fi

if [ $OPTION == "selected-models" ]
then 
    sudo docker run --gpus all -v /home/hcsuser/repos/onnxruntime/onnxruntime/python/tools/tensorrt/perf/:/usr/share/perf/ $DOCKER_IMAGE /bin/bash /usr/share/perf/perf.sh -d /usr/share/perf/ -o selected-models -m $MODEL_PATH
fi
