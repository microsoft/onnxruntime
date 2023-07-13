FROM rocm/pytorch:rocm5.6_ubuntu20.04_py3.8_pytorch_2.0.1

RUN apt-get update -y && apt-get upgrade -y && apt-get autoremove -y && apt-get clean -y

WORKDIR /stage

# from rocm/pytorch's image, work around ucx's dlopen replacement conflicting with shared provider
RUN cd /opt/mpi_install/ucx/build &&\
      make clean &&\
      ../contrib/configure-release --prefix=/opt/ucx --without-rocm &&\
      make -j $(nproc) &&\
      make install

# CMake
ENV CMAKE_VERSION=3.26.3
RUN cd /usr/local && \
    wget -q -O - https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.tar.gz | tar zxf -
ENV PATH=/usr/local/cmake-${CMAKE_VERSION}-linux-x86_64/bin:${PATH}

# ccache
RUN mkdir -p /tmp/ccache && \
    cd /tmp/ccache && \
    wget -q -O - https://github.com/ccache/ccache/releases/download/v4.7.4/ccache-4.7.4-linux-x86_64.tar.xz | tar --strip 1 -J -xf - && \
    cp /tmp/ccache/ccache /usr/bin && \
    rm -rf /tmp/ccache

RUN apt-get update && apt-get install  -y cifs-utils

# rocm-ci branch contains instrumentation needed for loss curves and perf
RUN git clone https://github.com/microsoft/huggingface-transformers.git &&\
      cd huggingface-transformers &&\
      git checkout rocm-ci &&\
      pip install -e .

RUN pip install \
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
      dill==0.3.4 \
      wget \
      pytorch_lightning==1.6.0 \
      pytest-xdist \
      pytest-rerunfailures

RUN pip install torch-ort --no-dependencies
ENV ORTMODULE_ONNX_OPSET_VERSION=15

# Install Cupy to decrease CPU utilization
# Install non dev openmpi
RUN rm -rf /opt/ompi && \
    wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.bz2 && \
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

ARG BUILD_UID=1001
ARG BUILD_USER=onnxruntimedev
RUN adduser --uid $BUILD_UID $BUILD_USER
WORKDIR /home/$BUILD_USER
