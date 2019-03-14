#!/bin/bash
set -e -o -x

SCRIPT_DIR="$( dirname "${BASH_SOURCE[0]}" )"

while getopts c:d:x: parameter_Option
do case "${parameter_Option}"
in
d) BUILD_DEVICE=${OPTARG};;
x) BUILD_EXTR_PAR=${OPTARG};;
esac
done

if [ $BUILD_DEVICE = "gpu" ]; then
    _CUDNN_VERSION=$(echo $CUDNN_VERSION | cut -d. -f1-2)
    python3 $SCRIPT_DIR/../../build.py --build_dir /home/onnxruntimedev \
        --config Debug Release \
        --skip_submodule_sync --enable_onnx_tests\
        --parallel --build_shared_lib \
        --use_cuda \
        --cuda_home /usr/local/cuda \
        --cudnn_home /usr/local/cudnn-$_CUDNN_VERSION/cuda --build_shared_lib $BUILD_EXTR_PAR
elif [ $BUILD_DEVICE = "fpga" ]; then
    # Add or update nuget sources
    for sourceName in Bond MsAzure Catapult CatapultTest ONNXRuntime; do
      mono /usr/bin/nuget.exe sources remove -Name $sourceName || true
    done

    mono /usr/bin/nuget.exe sources add -Name Bond -Source https://mscatapult.pkgs.visualstudio.com/_packaging/Bond/nuget/v3/index.json -Username lotusdev -Password $MSCATAPULT_PAT
    mono /usr/bin/nuget.exe sources add -Name MsAzure -Source https://msazure.pkgs.visualstudio.com/_packaging/Official/nuget/v3/index.json -Username lotusdev -Password $MSAZURE_PAT
    mono /usr/bin/nuget.exe sources add -Name Catapult -Source https://mscatapult.pkgs.visualstudio.com/_packaging/Catapult/nuget/v3/index.json -Username lotusdev -Password $MSCATAPULT_PAT
    mono /usr/bin/nuget.exe sources add -Name CatapultTest -Source https://mscatapult.pkgs.visualstudio.com/_packaging/testFeed/nuget/v3/index.json -Username lotusdev -Password $MSCATAPULT_PAT
    mono /usr/bin/nuget.exe sources add -Name ONNXRuntime -Source https://aiinfra.pkgs.visualstudio.com/_packaging/Lotus/nuget/v3/index.json -Username lotusdev -Password $AIINFRA_PAT

    mono /usr/bin/nuget.exe restore /onnxruntime_src/packages.config -PackagesDirectory /onnxruntime_src/nuget_root -NoCache

    python3 $SCRIPT_DIR/../../build.py --build_dir /home/onnxruntimedev \
        --config Debug Release \
        --skip_submodule_sync \
        --build_shared_lib \
        --use_mkldnn \
        --use_brainslice \
        --brain_slice_package_path /onnxruntime_src/nuget_root \
        --brain_slice_package_name CatapultFpga.Linux.5.1.3.40 \
        --brain_slice_client_package_name BrainSlice.v3.Client.3.0.0 \
        --parallel --enable_msinternal $BUILD_EXTR_PAR
else
    python3 $SCRIPT_DIR/../../build.py --build_dir /home/onnxruntimedev \
        --config Debug Release --build_shared_lib \
        --skip_submodule_sync --enable_onnx_tests \
        --enable_pybind \
        --parallel --build_shared_lib --enable_msinternal $BUILD_EXTR_PAR
fi
rm -rf /home/onnxruntimedev/models
