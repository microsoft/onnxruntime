# This docker file is for building with ib-enabled ompi libraries
ARG OPENMPI_VERSIONBASE=3.1
ARG OPENMPI_VERSIONMINOR=2
ARG OPENMPI_VERSION=${OPENMPI_VERSIONBASE}.${OPENMPI_VERSIONMINOR}
FROM onnxtraining.azurecr.io/azureml/base-gpu:openmpi${OPENMPI_VERSION}-cuda10.0-cudnn7-ubuntu16.04

# Set ARGs again since FROM resets all ARGS...
ARG OPENMPI_VERSIONBASE=3.1
ARG OPENMPI_VERSIONMINOR=2
ARG OPENMPI_VERSION=${OPENMPI_VERSIONBASE}.${OPENMPI_VERSIONMINOR}

# IB libs
RUN apt-get update &&      apt-get install -y --no-install-recommends      libmlx4-1      libmlx5-1      librdmacm1      libibverbs1      libmthca1      libdapl2      dapl2-utils      openssh-client      openssh-server      iproute2
RUN apt-get install -y --no-install-recommends libnuma-dev
RUN apt-get install -y --no-install-recommends cpio libibverbs1 libmlx4-1 libmlx5-1 librdmacm1 libmthca1 libdapl2 dapl2-utils pciutils ibutils ibverbs-utils rdmacm-utils infiniband-diags perftest librdmacm-dev libibverbs-dev
# Install ucx so that ompi can use IB directly
RUN wget https://github.com/openucx/ucx/releases/download/v1.6.1-rc2/ucx-1.6.1.tar.gz && tar zxf ucx-1.6.1.tar.gz && cd ucx-1.6.1 && ./configure --prefix=/usr/local --enable-optimizations --disable-assertions --disable-params-check --enable-mt && make -j $(nproc --all) && make install
# OMPI
RUN wget https://download.open-mpi.org/release/open-mpi/v${OPENMPI_VERSIONBASE}/openmpi-${OPENMPI_VERSION}.tar.gz && tar zxf openmpi-${OPENMPI_VERSION}.tar.gz && cd openmpi-${OPENMPI_VERSION} && ./configure --with-ucx=/usr/local/ --enable-mca-no-build=btl-uct --prefix=/usr/local/openmpi-${OPENMPI_VERSION}/ && make -j $(nproc --all) && make -j"$(nproc)" install
RUN ln -fs /usr/local/openmpi-${OPENMPI_VERSION} /usr/local/mpi
# Test installation
RUN test -f /usr/local/openmpi-${OPENMPI_VERSION}/bin/mpic++
ENV LD_LIBRARY_PATH=/usr/local/openmpi-${OPENMPI_VERSION}/lib:/usr/local/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV PATH=/usr/local/openmpi-${OPENMPI_VERSION}/bin:$PATH

WORKDIR /workspace

COPY onnxruntime_training_bert /workspace