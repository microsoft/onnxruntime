#!/bin/bash
set -e -x

INSTALL_DEPS_TRAINING=false
INSTALL_DEPS_DISTRIBUTED_SETUP=false
ORTMODULE_BUILD=false
TARGET_ROCM=false
USE_CONDA=false

while getopts p:h:d:v:tmurc parameter_Option
do case "${parameter_Option}"
in
p) PYTHON_VER=${OPTARG};;
h) TORCH_VERSION=${OPTARG};;
d) DEVICE_TYPE=${OPTARG};;
v) CU_VER=${OPTARG};;
t) INSTALL_DEPS_TRAINING=true;;
m) INSTALL_DEPS_DISTRIBUTED_SETUP=true;;
u) ORTMODULE_BUILD=true;;
r) TARGET_ROCM=true;;
c) USE_CONDA=true;;
esac
done

echo "Python version=$PYTHON_VER"

DEVICE_TYPE=${DEVICE_TYPE:=Normal}

if [[ $USE_CONDA = true ]]; then
  # conda python version has already been installed by
  # tools/ci_build/github/linux/docker/Dockerfile.ubuntu_gpu_training.
  # so, /home/onnxruntimedev/miniconda3/bin/python should point
  # to the correct version of the python version
   PYTHON_EXE="/home/onnxruntimedev/miniconda3/bin/python"
elif [[ "$PYTHON_VER" = "3.6" && -d "/opt/python/cp36-cp36m"  ]]; then
   PYTHON_EXE="/opt/python/cp36-cp36m/bin/python3.6"
elif [[ "$PYTHON_VER" = "3.7" && -d "/opt/python/cp37-cp37m"  ]]; then
   PYTHON_EXE="/opt/python/cp37-cp37m/bin/python3.7"
elif [[ "$PYTHON_VER" = "3.8" && -d "/opt/python/cp38-cp38"  ]]; then
   PYTHON_EXE="/opt/python/cp38-cp38/bin/python3.8"
elif [[ "$PYTHON_VER" = "3.9" && -d "/opt/python/cp39-cp39"  ]]; then
   PYTHON_EXE="/opt/python/cp39-cp39/bin/python3.9"
else
   PYTHON_EXE="/usr/bin/python${PYTHON_VER}"
fi

export ONNX_ML=1
export CMAKE_ARGS="-DONNX_GEN_PB_TYPE_STUBS=OFF -DONNX_WERROR=OFF"
${PYTHON_EXE} -m pip install -r ${0/%install_python_deps_eager\.sh/requirements\.txt}
${PYTHON_EXE} -m pip install -r ${0/%install_python_deps_eager.sh/training\/ortmodule\/stage1\/requirements_torch_eager_cpu_ubuntu.txt}


# INSTALL_DEPS_TRAINING=false
# INSTALL_DEPS_DISTRIBUTED_SETUP=false
# ORTMODULE_BUILD=false
# TARGET_ROCM=false
# USE_CONDA=false

# while getopts p:h:d:v:tmurc parameter_Option
# do case "${parameter_Option}"
# in
# p) PYTHON_VER=${OPTARG};;
# h) TORCH_VERSION=${OPTARG};;
# d) DEVICE_TYPE=${OPTARG};;
# v) CU_VER=${OPTARG};;
# t) INSTALL_DEPS_TRAINING=true;;
# m) INSTALL_DEPS_DISTRIBUTED_SETUP=true;;
# u) ORTMODULE_BUILD=true;;
# r) TARGET_ROCM=true;;
# c) USE_CONDA=true;;
# esac
# done

# echo "Python version=$PYTHON_VER"

# DEVICE_TYPE=${DEVICE_TYPE:=Normal}

# if [[ $USE_CONDA = true ]]; then
#   # conda python version has already been installed by
#   # tools/ci_build/github/linux/docker/Dockerfile.ubuntu_gpu_training.
#   # so, /home/onnxruntimedev/miniconda3/bin/python should point
#   # to the correct version of the python version
#    PYTHON_EXE="/home/onnxruntimedev/miniconda3/bin/python"
# elif [[ "$PYTHON_VER" = "3.6" && -d "/opt/python/cp36-cp36m"  ]]; then
#    PYTHON_EXE="/opt/python/cp36-cp36m/bin/python3.6"
# elif [[ "$PYTHON_VER" = "3.7" && -d "/opt/python/cp37-cp37m"  ]]; then
#    PYTHON_EXE="/opt/python/cp37-cp37m/bin/python3.7"
# elif [[ "$PYTHON_VER" = "3.8" && -d "/opt/python/cp38-cp38"  ]]; then
#    PYTHON_EXE="/opt/python/cp38-cp38/bin/python3.8"
# elif [[ "$PYTHON_VER" = "3.9" && -d "/opt/python/cp39-cp39"  ]]; then
#    PYTHON_EXE="/opt/python/cp39-cp39/bin/python3.9"
# else
#    PYTHON_EXE="/usr/bin/python${PYTHON_VER}"
# fi

# export ONNX_ML=1
# export CMAKE_ARGS="-DONNX_GEN_PB_TYPE_STUBS=OFF -DONNX_WERROR=OFF"
# ${PYTHON_EXE} -m pip install -r ${0/%install_python_deps_eager\.sh/requirements\.txt}
# ${PYTHON_EXE} -m pip install -r ${0/%install_python_deps_eager.sh/training\/ortmodule\/stage1\/requirements_torch_cpu.txt}
