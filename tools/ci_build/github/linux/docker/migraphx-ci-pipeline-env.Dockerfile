ARG ROCM_VERSION=5.6

FROM rocm/dev-ubuntu-22.04:${ROCM_VERSION}-complete

# Upgrade to meet security requirements
RUN apt-get update -y && apt-get upgrade -y && apt-get autoremove -y && \
    apt-get install  -y locales cifs-utils wget half libnuma-dev lsb-release && \
    apt-get clean -y

ENV DEBIAN_FRONTEND noninteractive
ENV MIGRAPHX_DISABLE_FAST_GELU=1
RUN locale-gen en_US.UTF-8
RUN update-locale LANG=en_US.UTF-8
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

WORKDIR /stage

# Cmake
ENV CMAKE_VERSION=3.26.3
RUN cd /usr/local && \
    wget -q https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.tar.gz && \
    tar -zxf /usr/local/cmake-3.26.3-Linux-x86_64.tar.gz --strip=1 -C /usr

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

# Conda base patch
RUN pip install cryptography==41.0.0

# Create migraphx-ci environment
ENV CONDA_ENVIRONMENT_PATH /opt/miniconda/envs/migraphx-ci
ENV CONDA_DEFAULT_ENV migraphx-ci
RUN conda create -y -n ${CONDA_DEFAULT_ENV} python=3.8
ENV PATH ${CONDA_ENVIRONMENT_PATH}/bin:${PATH}

# Enable migraphx-ci environment
SHELL ["conda", "run", "-n", "migraphx-ci", "/bin/bash", "-c"]

# Install migraphx
RUN apt update && apt install -y migraphx

RUN pip install numpy packaging
