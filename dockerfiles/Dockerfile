FROM ubuntu:16.04

RUN apt-get update && apt-get install -y sudo build-essential curl libcurl4-openssl-dev libssl-dev wget python3 python3-pip python3-dev git 
RUN pip3 install numpy onnx

# Build the latest cmake 
WORKDIR /code 
RUN wget https://cmake.org/files/v3.12/cmake-3.12.3.tar.gz; 
RUN tar zxf cmake-3.12.3.tar.gz

WORKDIR /code/cmake-3.12.3 
RUN ./configure --system-curl 
RUN make 
RUN sudo make install

RUN pip3 install onnxruntime jupyter
WORKDIR /code