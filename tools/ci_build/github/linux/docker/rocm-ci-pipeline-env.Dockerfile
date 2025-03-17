# Refer to https://github.com/RadeonOpenCompute/ROCm-docker/blob/master/dev/Dockerfile-ubuntu-22.04-complete
FROM ubuntu:22.04

ARG ROCM_VERSION=6.1.3
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
    apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    libelf1 \
    kmod \
    file \
    python3 \
    python3-pip \
    rocm-dev \
    rocm-libs \
    build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd -g 109 render

# Upgrade to meet security requirements
RUN apt-get update -y && apt-get upgrade -y && apt-get autoremove -y && \
    apt-get install  -y locales cifs-utils wget half libnuma-dev lsb-release && \
    apt-get clean -y

RUN locale-gen en_US.UTF-8
RUN update-locale LANG=en_US.UTF-8
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

WORKDIR /stage

# Cmake
ENV CMAKE_VERSION=3.30.1
RUN cd /usr/local && \
    wget -q https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.tar.gz && \
    tar -zxf /usr/local/cmake-3.30.1-Linux-x86_64.tar.gz --strip=1 -C /usr

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
RUN conda create -y -n ${CONDA_DEFAULT_ENV} python=3.10
ENV PATH ${CONDA_ENVIRONMENT_PATH}/bin:${PATH}

# Enable rocm-ci environment
SHELL ["conda", "run", "-n", "rocm-ci", "/bin/bash", "-c"]

# Some DLLs in the conda environment have conflict with the one installed in Ubuntu system.
# For example, the GCC version in the conda environment is 12.x, while the one in the Ubuntu 22.04 is 11.x.
# ln -sf to make sure we always use libstdc++.so.6 and libgcc_s.so.1 in the system.
RUN ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ${CONDA_ENVIRONMENT_PATH}/bin/../lib/libstdc++.so.6
RUN ln -sf /usr/lib/x86_64-linux-gnu/libgcc_s.so.1 ${CONDA_ENVIRONMENT_PATH}/bin/../lib/libgcc_s.so.1

RUN pip install packaging \
                ml_dtypes==0.5.0 \
                pytest==7.4.4 \
                pytest-xdist \
                pytest-rerunfailures \
                scipy==1.14.1 \
                numpy==1.26.4

RUN apt install -y git

# Install Cupy to decrease CPU utilization
# Note that the version of Cupy requires numpy < 1.27
RUN git clone https://github.com/ROCm/cupy && cd cupy && \
    git checkout 432a8683351d681e00903640489cb2f4055d2e09 && \
    export CUPY_INSTALL_USE_HIP=1 && \
    export ROCM_HOME=/opt/rocm && \
    export HCC_AMDGPU_TARGET=gfx906,gfx908,gfx90a && \
    git submodule update --init && \
    pip install -e . --no-cache-dir -vvvv
