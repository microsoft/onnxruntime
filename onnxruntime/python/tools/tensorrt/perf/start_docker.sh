#!/bin/bash

sudo systemctl start docker.service    
sudo docker run --gpus all -it -v /home/hcsuser/:/usr/share/ 36f6f9268db2
cd /usr/share/repos/onnxruntime/onnxruntime/python/tools/tensorrt/perf
