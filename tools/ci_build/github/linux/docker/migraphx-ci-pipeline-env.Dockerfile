FROM rocm/pytorch:rocm5.5_ubuntu20.04_py3.8_pytorch_1.13.1

# MIGraphX version should be the same as ROCm version
ARG MIGRAPHX_VERSION=rocm-5.5.0

ENV DEBIAN_FRONTEND noninteractive
ENV MIGRAPHX_DISABLE_FAST_GELU=1

RUN apt-get update -y && apt-get upgrade -y && apt-get autoremove -y libprotobuf\* protobuf-compiler\* && \
    apt-get install -y locales unzip && apt-get clean -y
RUN locale-gen en_US.UTF-8
RUN update-locale LANG=en_US.UTF-8
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

WORKDIR /stage

ADD scripts /tmp/scripts
RUN /tmp/scripts/install_os_deps.sh -p /usr/local && rm -rf /tmp/scripts

# from rocm/pytorch's image, work around ucx's dlopen replacement conflicting with shared provider
RUN cd /opt/mpi_install/ucx/build &&\
      make clean &&\
      ../contrib/configure-release --prefix=/opt/ucx --without-rocm &&\
      make -j $(nproc) &&\
      make install

RUN apt-get update &&\
    apt-get install -y half libnuma-dev

# Install rbuild
RUN pip3 install https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz numpy yapf==0.28.0

# Install MIGraphX from source
RUN mkdir -p /migraphx
RUN cd /migraphx && git clone --depth=1 --branch ${MIGRAPHX_VERSION} https://github.com/ROCmSoftwarePlatform/AMDMIGraphX src
RUN cd /migraphx && rbuild package --cxx /opt/rocm/llvm/bin/clang++ -d /migraphx/deps -B /migraphx/build -S /migraphx/src/ -DPYTHON_EXECUTABLE=/usr/bin/python3
RUN dpkg -i /migraphx/build/*.deb
RUN rm -rf /migraphx

# ccache
RUN mkdir -p /tmp/ccache && \
    cd /tmp/ccache && \
    wget -q -O - https://github.com/ccache/ccache/releases/download/v4.7.4/ccache-4.7.4-linux-x86_64.tar.xz | tar --strip 1 -J -xf - && \
    cp /tmp/ccache/ccache /usr/bin && \
    rm -rf /tmp/ccache
