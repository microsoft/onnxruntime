set -x

TMP_DIR=/tmp/onnxruntime

# update dir
# CHANGED_FILE=onnxruntime/core/providers/rocm/cu_inc/common.cuh
# cp -r $CHANGED_FILE $TMP_DIR/$CHANGED_FILE
cp -r . $TMP_DIR

# swith to tmp dir
cd $TMP_DIR
pwd
ls

# ./build.sh --build_dir build --config RelWithDebInfo --parallel --use_hip --hip_home /opt/rocm --enable_training
# ./build.sh --build_dir build --config Release --use_migraphx --migraphx_home  /opt/rocm
export ROCM_HOME=/opt/rocm
HIP_HIPCC_FLAGS=--save-temps python tools/ci_build/build.py \
    --config RelWithDebInfo \
    --enable_training \
    --mpi_home /opt/ompi \
    --use_rocm \
    --rocm_version=4.5.2 \
    --rocm_home /opt/rocm \
    --nccl_home /opt/rocm \
    --update \
    --build_dir ./build \
    --build \
    --parallel 8 \
    --build_wheel \
    --skip_tests
