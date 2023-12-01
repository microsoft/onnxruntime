#!/bin/bash
echo -e "\033[31m you can edit the `realpath $0` as you need \033[0m"

branch=zhijxu/code-reading
build_type=Debug
cuda=11.8
dev_opts="--enable_nvtx_profile --enable_cuda_line_info"


set -ex
# exit if $1 is not "build" or "update"
if [[ $1 != "build" &&  $1 != "update" ]]
then
    echo -e "\033[31m you need to specify build or update \033[0m"
    exit 1
fi

output_dir=/tmp/ort_build/$branch/$build_type
# in case you use "root" to build
sudo git config --system --add safe.directory "*"

cd ..
git checkout $branch
if [[ $branch != `git branch | grep '*' | awk '{print $2}' -` ]]
then
    echo branch is wrong
    exit 666 # remove the exit if you want to continue to build
fi

if [ $1 == "build" ]
then
    echo -e "\033[32m start building \033[0m"
    sudo rm -rf $output_dir/$build_type/dist/*.whl

    bash ./build.sh \
        --build_dir=$output_dir \
        --cuda_home /usr/local/cuda-$cuda --cudnn_home /usr/lib/x86_64-linux-gnu/ \
        --cuda_version=$cuda \
        --use_cuda --config $build_type --update --build \
        --build_wheel \
        --parallel \
        --enable_training --skip_tests \
        --cmake_extra_defines ONNXRUNTIME_VERSION=$(cat ./VERSION_NUMBER) CMAKE_CUDA_ARCHITECTURES="70;75" onnxruntime_BUILD_UNIT_TESTS=OFF \
        --use_mpi=false $dev_opts

    sudo env "PATH=$PATH" pip uninstall -y onnxruntime-training
    sudo env "PATH=$PATH" pip uninstall -y onnxruntime

    sudo env "PATH=$PATH" pip install $output_dir/$build_type/dist/*.whl
    cd /tmp
    sudo env "PATH=$PATH"  TORCH_CUDA_ARCH_LIST="7.0+PTX" python -m torch_ort.configure
    echo -e "\033[0;31m build and install ORT from scratch done\033[0m"
else
    echo -e "\033[0;31m update ort package \033[0m"
    build_path=$(realpath -s $output_dir/$build_type)
    cd $build_path
    # VERBOSE=1
    make -j40 onnxruntime_pybind11_state
    cd /tmp
    wheel_path=`pip show onnxruntime-training | grep -i location | cut  -d" " -f2`
    cd $wheel_path/onnxruntime/capi
    so_files=`ls *.so`
    sudo rm -rf $so_files
    for i in $so_files
    do
        sudo ln -s  $build_path/$i $i
    done
    echo -e "\033[0;31m update ORT by softlink done \033[0m"
fi
