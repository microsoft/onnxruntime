# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# multi-stage arguments (repeat ARG NAME below)
ARG MOFED_VERSION=4.7-3.2.9.0
ARG UCX_VERSION=1.8.0
ARG OPENMPI_VERSION=4.0.4
ARG CONDA_VERSION=4.7.10
ARG MKL_VERSION=2020.1
ARG NUMPY_VERSION=1.18.5
ARG ONNX_VERSION=1.7.0

# cuda development image for building sources
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04 as build
RUN apt-get -y update

# set location for builds
WORKDIR /stage
    
# install miniconda (comes with python 3.7 default)
ARG CONDA_VERSION
ARG CONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-x86_64.sh
RUN cd /stage && curl -fSsL --insecure ${CONDA_URL} -o install-conda.sh &&\
    /bin/bash ./install-conda.sh -b -p /opt/conda &&\
    /opt/conda/bin/conda clean -ya
ENV PATH=/opt/conda/bin:${PATH}

# install mkl, numpy, and pytorch build dependencies
ARG MKL_VERSION
ARG NUMPY_VERSION
ARG ONNX_VERSION
RUN conda install -y \
        mkl=${MKL_VERSION} \
        mkl-include=${MKL_VERSION} \
        numpy=${NUMPY_VERSION} \
        cmake \
        ninja \
        pyyaml \
        cffi \
        setuptools &&\
    pip install \
        onnx=="${ONNX_VERSION}"

# install mellanox libraries for ib etc.
ARG MOFED_VERSION
ARG MOFED_OS=ubuntu18.04
ARG MOFED_TARNAME=MLNX_OFED_LINUX-${MOFED_VERSION}-${MOFED_OS}-x86_64
ARG MOFED_URL=https://www.mellanox.com/downloads/ofed/MLNX_OFED-${MOFED_VERSION}/${MOFED_TARNAME}.tgz
RUN cd /stage && curl -fSsL ${MOFED_URL} | tar -zxpf - &&\
    mv ${MOFED_TARNAME} mofed-${MOFED_VERSION} &&\
    cd mofed-${MOFED_VERSION} && \
    ./mlnxofedinstall --basic --force --user-space-only --without-fw-update

# build ucx suite (leverages mellanox libraries)
# note: openmpi will not select ucx without multithreading enabled
ARG UCX_VERSION
ARG UCX_TARNAME=ucx-$UCX_VERSION
ARG UCX_URL=https://github.com/openucx/ucx/releases/download/v${UCX_VERSION}/${UCX_TARNAME}.tar.gz
RUN apt-get -y install \
        libnuma-dev &&\
    cd /stage && curl -fSsL ${UCX_URL} | tar xzf - &&\
    cd ${UCX_TARNAME} &&\
    ./configure \
	--prefix=/opt/ucx \
        --with-cuda=/usr/local/cuda \
        --with-verbs=/usr \
        --enable-mt &&\
    make -j"$(nproc)" &&\
    make install

# build openmpi (use --prefix /opt/ompi4 to move to runtime image)
# note: require --enable-orterun-prefix-by-default for Azure machine learning compute
# note: disable verbs as we use ucx middleware and don't want btl openib warnings
ARG OPENMPI_VERSION
ARG OPENMPI_TARNAME=openmpi-${OPENMPI_VERSION}
ARG OPENMPI_URL=https://download.open-mpi.org/release/open-mpi/v%OMPI_BASE%/${OPENMPI_TARNAME}.tar.gz
RUN export OMPI_BASE=${OPENMPI_VERSION%.*} &&\
    cd /stage && curl -fSsL `echo ${OPENMPI_URL} | sed s/%OMPI_BASE%/$OMPI_BASE/` | tar xzf - &&\
    cd ${OPENMPI_TARNAME} &&\
    ./configure \
        --prefix=/opt/ompi4 \
        --with-ucx=/opt/ucx \
        --without-verbs \
        --with-cuda=/usr/local/cuda \
        --enable-mpirun-prefix-by-default \
        --enable-orterun-prefix-by-default \
        --enable-mca-no-build=btl-uct &&\
    make -j"$(nproc)" install &&\
    ldconfig
ENV PATH=/opt/ompi4/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/ompi4/lib:$LD_LIBRARY_PATH

# install mpi4py (be sure to link existing /opt/ompi4)
RUN CC=mpicc MPICC=mpicc pip install mpi4py --no-binary mpi4py

# build pytorch (using v1.6.0-rc3 + patch)
# todo: need to update to pytorch 1.6.0 releaase when ready
# note: FindNCCL.patch is in pytorch master
COPY FindNCCL.patch /stage/
ENV CUDA_HOME="/usr/local/cuda" \
    CUDNN_LIBRARY="/usr/lib/x86_64-linux-gnu" \
    NCCL_INCLUDE_DIR="/usr/local/cuda/include" \
    NCCL_LIB_DIR="/usr/local/cuda/lib64" \
    USE_SYSTEM_NCCL=1 
RUN apt-get -y install git &&\
    cd /stage && git clone https://github.com/pytorch/pytorch.git &&\
    cd pytorch &&\
    git checkout v1.6.0-rc3 &&\
    patch cmake/Modules/FindNCCL.cmake /stage/FindNCCL.patch &&\
    git submodule update --init --recursive &&\
    python setup.py bdist_wheel -d build/wheel &&\
    pip install build/wheel/*.whl

# build onnxruntime wheel with cuda and mpi support
RUN cd /stage && git clone https://github.com/microsoft/onnxruntime.git &&\
    cd onnxruntime &&\
    git checkout v1.4.0 &&\
    python tools/ci_build/build.py \
        --cmake_extra_defines \
            ONNXRUNTIME_VERSION=`cat ./VERSION_NUMBER` \
        --config RelWithDebInfo \
        --enable_training \
        --mpi_home /opt/ompi4 \
        --use_cuda \
        --cuda_home /usr/local/cuda \
        --cudnn_home /usr/lib/x86_64-linux-gnu/ \
        --nccl_home /usr/local/cuda/targets/x86_64-linux \
        --update \
        --parallel \
        --build_dir build \
        --build \
        --build_wheel \
        --skip_tests &&\
    pip install build/RelWithDebInfo/dist/*.whl

# switch to cuda runtime environment
# note: launch with --gpus all or nvidia-docker
FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04
RUN apt-get -y update
WORKDIR /stage

# install mellanox libraries (again)
# note: launch with infiniband drivers and --net=host or docker swarm
ARG MOFED_VERSION
COPY --from=build /stage/mofed-${MOFED_VERSION} /stage/mofed-${MOFED_VERSION}
RUN apt-get install -y build-essential &&\
    cd /stage/mofed-${MOFED_VERSION} &&\
    ./mlnxofedinstall --basic --force --user-space-only --without-fw-update

# install ucx
# note: launch with --cap-add=sys_nice to avoid 'mbind' warnings
COPY --from=build /opt/ucx /opt/ucx
ENV PATH=/opt/ucx/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/ucx/lib:$LD_LIBRARY_PATH

# install openmpi
# note: permit mpirun as root for Azure cluster submissions
# note: enforce openmpi select ucx or fail
COPY --from=build /opt/ompi4 /opt/ompi4
ENV PATH=/opt/ompi4/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/ompi4/lib:$LD_LIBRARY_PATH
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ENV OMPI_MCA_pml=ucx
RUN apt-get install -y \
        openssh-server \
        openssh-client \
        libnuma-dev &&\
    ldconfig

# copy conda environment (includes mkl, numpy, pytorch, onnxruntime)
COPY --from=build /opt/conda /opt/conda
ENV PATH=/opt/conda/bin:${PATH}

# make ssh/sshd less strict for wiring containers on Azure VM scale set
# note: use 'service ssh start' to launch sshd (will fail if 22 in use)
# note: can also set port != 22 and set port=X in MPI hosts file
# note: need to setup password free ssh login between MPI hosts
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/g' \
        /etc/ssh/sshd_config &&\
    sed -i 's/#StrictModes yes/StrictModes no/g' \
        /etc/ssh/sshd_config &&\
    sed -i 's/#   StrictHostKeyChecking ask/    StrictHostKeyChecking no/g' \
        /etc/ssh/ssh_config &&\
    mkdir /run/sshd &&\
    chmod u=rwx,go=rx /run/sshd

# clean\finalize environment
RUN conda remove -y cmake ninja &&\
    apt-get remove -y build-essential &&\
    apt-get autoremove -y &&\
    rm -fr /stage
WORKDIR /workspace
