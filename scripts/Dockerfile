FROM ubuntu:18.04
RUN apt-get update && \
    apt-get install -y sudo git bash python3-pip && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install tensorflow==1.15 tf2onnx onnx tf_slim
    