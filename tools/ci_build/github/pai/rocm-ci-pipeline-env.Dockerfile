FROM rocm/pytorch:rocm5.4_ubuntu20.04_py3.7_pytorch_1.12.1

WORKDIR /stage

# from rocm/pytorch's image, work around ucx's dlopen replacement conflicting with shared provider
RUN cd /opt/mpi_install/ucx/build &&\
      make clean &&\
      ../contrib/configure-release --prefix=/opt/ucx --without-rocm &&\
      make -j $(nproc) &&\
      make install

# CMake
ENV CMAKE_VERSION=3.24.2
RUN cd /usr/local && \
    wget -q -O - https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.tar.gz | tar zxf -
ENV PATH=/usr/local/cmake-${CMAKE_VERSION}-linux-x86_64/bin:${PATH}

# ccache
RUN mkdir -p /tmp/ccache && \
    cd /tmp/ccache && \
    wget -q -O - https://github.com/ccache/ccache/releases/download/v4.7.4/ccache-4.7.4-linux-x86_64.tar.xz | tar --strip 1 -J -xf - && \
    cp /tmp/ccache/ccache /usr/bin && \
    rm -rf /tmp/ccache

# rocm-ci branch contains instrumentation needed for loss curves and perf
RUN git clone https://github.com/microsoft/huggingface-transformers.git &&\
      cd huggingface-transformers &&\
      git checkout rocm-ci &&\
      pip install -e .

RUN pip install \
      numpy \
      onnx \
      cerberus \
      sympy \
      h5py \
      datasets==1.9.0 \
      requests \
      sacrebleu==1.5.1 \
      sacremoses \
      scipy \
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
