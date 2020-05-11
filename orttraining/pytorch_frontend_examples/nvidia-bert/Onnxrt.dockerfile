# FROM nvcr.io/nvidia/tritonserver:20.03-py3-clientsdk as trt
FROM mcr.microsoft.com/azureml/base-gpu:openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:/opt/cmake/bin:/opt/miniconda/bin:${PATH}
ENV LD_LIBRARY_PATH /opt/miniconda/lib:$LD_LIBRARY_PATH

RUN apt update -y && pip install --upgrade pip

# install pytorch stable using CUDA 10.1
RUN pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# add pytorch patch for opset 10
COPY pyt_patch /tmp/pyt_patch
RUN cp /tmp/pyt_patch/symbolic_opset10.py /opt/miniconda/lib/python3.7/site-packages/torch/onnx/

# install onnxruntime wheel built off master 05/01/2020
# COPY ort_wheel /tmp/ort_wheel
# RUN pip install /tmp/ort_wheel/onnxruntime_gpu-master05012020-cp37-cp37m-linux_x86_64.whl && rm -r /tmp/ort_wheel

# build onnxruntime from source
WORKDIR /src
RUN wget --quiet https://github.com/Kitware/CMake/releases/download/v3.14.3/cmake-3.14.3-Linux-x86_64.tar.gz &&\
    tar zxf cmake-3.14.3-Linux-x86_64.tar.gz &&\
    mv cmake-3.14.3-Linux-x86_64 /opt/cmake &&\
    rm -rf cmake-3.14.3-Linux-x86_64.tar.gz

RUN git clone https://github.com/microsoft/onnxruntime.git &&\
    cd /src/onnxruntime &&\
    git checkout orttraining_rc1 &&\
    /bin/sh ./build.sh \
        --config RelWithDebInfo \
        --use_cuda \
        --cuda_home /usr/local/cuda \
        --cudnn_home /usr/lib/x86_64-linux-gnu/ \
        --update \
        --build \
        --build_wheel \
        --enable_training \
        --parallel \
        --cmake_extra_defines ONNXRUNTIME_VERSION=`cat ./VERSION_NUMBER`
RUN pip install \
          /src/onnxruntime/build/Linux/RelWithDebInfo/dist/*.whl

# build and install apex
RUN git clone https://github.com/NVIDIA/apex && cd apex &&\
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# below is similar to NVIDIA dockerfile
# --- 

RUN apt-get update && apt-get install -y pbzip2 pv bzip2 cabextract

ENV BERT_PREP_WORKING_DIR /workspace/bert/data

WORKDIR /workspace
# RUN git clone https://github.com/attardi/wikiextractor.git
# RUN git clone https://github.com/soskek/bookcorpus.git

# Copy the perf_client over
# COPY --from=trt /workspace/install/ /workspace/install/
# ENV LD_LIBRARY_PATH /workspace/install/lib:${LD_LIBRARY_PATH}

# Install trt python api
# RUN pip install /workspace/install/python/tensorrtserver-1.*-py3-none-linux_x86_64.whl

WORKDIR /workspace/bert
RUN pip install --upgrade --no-cache-dir pip \
 && pip install --no-cache-dir \
 tqdm boto3 requests six ipdb h5py html2text nltk progressbar mpi4py \
 git+https://github.com/NVIDIA/dllogger

RUN apt-get install -y iputils-ping

COPY . .
