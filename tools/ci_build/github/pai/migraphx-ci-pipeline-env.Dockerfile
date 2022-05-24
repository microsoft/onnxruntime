FROM rocm/pytorch:rocm5.1.1_ubuntu20.04_py3.7_pytorch_1.10.0

ENV DEBIAN_FRONTEND noninteractive
ENV MIGRAPHX_DISABLE_FAST_GELU=1

RUN apt-get clean && apt-get update && apt-get install -y locales
RUN locale-gen en_US.UTF-8
RUN update-locale LANG=en_US.UTF-8
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

WORKDIR /stage

# from rocm/pytorch's image, work around ucx's dlopen replacement conflicting with shared provider
RUN cd /opt/mpi_install/ucx/build &&\
      make clean &&\
      ../contrib/configure-release --prefix=/opt/ucx --without-rocm &&\
      make -j $(nproc) &&\
      make install

# CMake
ENV CMAKE_VERSION=3.18.2
RUN cd /usr/local && \
    wget -q -O - https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.tar.gz | tar zxf -
ENV PATH=/usr/local/cmake-${CMAKE_VERSION}-Linux-x86_64/bin:${PATH}

RUN apt-get update &&\
    apt-get install -y sudo git bash build-essential rocm-dev python3.7 python3-pip miopen-hip \
    rocblas half libnuma-dev

# Install rbuild
RUN pip3 install https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz numpy yapf==0.28.0

# Install MIGraphX from source
RUN mkdir -p /migraphx
RUN cd /migraphx && git clone --depth=1 --branch migraphx_for_ort https://github.com/ROCmSoftwarePlatform/AMDMIGraphX src
RUN cd /migraphx && rbuild package --cxx /opt/rocm-5.1.1/llvm/bin/clang++ -d /migraphx/deps -B /migraphx/build -S /migraphx/src/ -DPYTHON_EXECUTABLE=/usr/bin/python3
RUN dpkg -i /migraphx/build/*.deb
RUN rm -rf /migraphx
