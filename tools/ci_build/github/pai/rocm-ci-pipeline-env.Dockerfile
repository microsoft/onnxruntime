# Refer to https://github.com/RadeonOpenCompute/ROCm-docker/blob/master/dev/Dockerfile-ubuntu-22.04-complete
FROM ubuntu:22.04

ARG ROCM_VERSION=5.7
ARG AMDGPU_VERSION=${ROCM_VERSION}
ARG APT_PREF='Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600'

CMD ["/bin/bash"]

RUN echo "$APT_PREF" > /etc/apt/preferences.d/rocm-pin-600

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl libnuma-dev gnupg && \
    curl -sL https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add -   &&\
    printf "deb [arch=amd64] https://repo.radeon.com/rocm/apt/$ROCM_VERSION/ jammy main" | tee /etc/apt/sources.list.d/rocm.list   && \
    printf "deb [arch=amd64] https://repo.radeon.com/amdgpu/$AMDGPU_VERSION/ubuntu jammy main" | tee /etc/apt/sources.list.d/amdgpu.list   && \
    apt-get update && apt-get install -y --no-install-recommends  \
    sudo   \
    libelf1   \
    kmod   \
    file   \
    python3   \
    python3-pip   \
    rocm-dev   \
    rocm-libs   \
    build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd -g 109 render

RUN apt-get update -y && apt-get upgrade -y && apt-get autoremove -y libprotobuf\* protobuf-compiler\* && \
    rm -f /usr/local/bin/protoc && apt-get install -y locales unzip wget git && apt-get clean -y
RUN locale-gen en_US.UTF-8
RUN update-locale LANG=en_US.UTF-8
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

WORKDIR /stage

# CMake
ENV CMAKE_VERSION=3.27.3
RUN cd /usr/local && \
    wget -q -O - https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.tar.gz | tar zxf -
ENV PATH=/usr/local/cmake-${CMAKE_VERSION}-linux-x86_64/bin:${PATH}

# ccache
RUN mkdir -p /tmp/ccache && \
    cd /tmp/ccache && \
    wget -q -O - https://github.com/ccache/ccache/releases/download/v4.7.4/ccache-4.7.4-linux-x86_64.tar.xz | tar --strip 1 -J -xf - && \
    cp /tmp/ccache/ccache /usr/bin && \
    rm -rf /tmp/ccache

# Install Conda
ENV PATH /opt/miniconda/bin:${PATH}
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh --no-check-certificate && /bin/bash ~/miniconda.sh -b -p /opt/miniconda && \
    conda init bash && \
    conda config --set auto_activate_base false && \
    conda update --all && \
    rm ~/miniconda.sh && conda clean -ya

# Create rocm-ci environment
ENV CONDA_ENVIRONMENT_PATH /opt/miniconda/envs/rocm-ci
ENV CONDA_DEFAULT_ENV rocm-ci
RUN conda create -y -n ${CONDA_DEFAULT_ENV} python=3.9
ENV PATH ${CONDA_ENVIRONMENT_PATH}/bin:${PATH}

# Conda base patch
RUN pip install cryptography==41.0.4

# Enable rocm-ci environment
SHELL ["conda", "run", "-n", "rocm-ci", "/bin/bash", "-c"]

# ln -sf is needed to make sure that version `GLIBCXX_3.4.30' is found
RUN ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ${CONDA_ENVIRONMENT_PATH}/bin/../lib/libstdc++.so.6

# Install Pytorch
RUN pip install torch==2.0.1 torchvision==0.15.2 -f https://repo.radeon.com/rocm/manylinux/rocm-rel-${ROCM_VERSION}/ && \
    pip install torch-ort --no-dependencies


##### Install Cupy to decrease CPU utilization
# Install non dev openmpi
RUN wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.bz2 && \
    tar -jxf openmpi-4.1.5.tar.bz2 && \
    cd openmpi-4.1.5 && \
    ./configure --prefix=/opt/ompi && \
    make -j4 all && \
    make install && \
    cd ../ && \
    rm -r openmpi-4.1.5 && \
    rm openmpi-4.1.5.tar.bz2

# Install CuPy, No stable version is available
RUN git clone https://github.com/ROCmSoftwarePlatform/cupy && cd cupy && \
    git checkout fc251a808037f8a2270860c2a23a683bfc0de43e && \
    export CUPY_INSTALL_USE_HIP=1 && \
    export ROCM_HOME=/opt/rocm && \
    export HCC_AMDGPU_TARGET=gfx906,gfx908,gfx90a && \
    git submodule update --init && \
    pip install -e . --no-cache-dir -vvvv

##### Install transformers to run tests
# rocm-ci branch contains instrumentation needed for loss curves and perf
RUN git clone https://github.com/microsoft/huggingface-transformers.git &&\
    cd huggingface-transformers &&\
    git checkout rocm-ci &&\
    pip install -e .

RUN pip install \
    flatbuffers==2.0 \
    numpy==1.24.1 \
    onnx \
    cerberus \
    sympy \
    h5py \
    datasets==1.9.0 \
    requests \
    sacrebleu==1.5.1 \
    sacremoses \
    scipy==1.10.0 \
    scikit-learn \
    tokenizers \
    sentencepiece \
    wget \
    dill==0.3.4 \
    pytorch_lightning==1.6.0 \
    pytest-xdist \
    pytest-rerunfailures

ENV ORTMODULE_ONNX_OPSET_VERSION=15

ARG BUILD_UID=1001
ARG BUILD_USER=onnxruntimedev
RUN adduser --uid $BUILD_UID $BUILD_USER
WORKDIR /home/$BUILD_USER
