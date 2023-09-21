export CUDA_HOME=/usr/local/cuda-12.2
export CUDNN_HOME=/usr/lib/x86_64-linux-gnu/
export CUDACXX=/usr/local/cuda-12.2/bin/nvcc
export TRT_HOME=/usr/src/tensorrt

sh build.sh --config Release  --build_shared_lib --parallel  --use_cuda --cuda_version 12.2 \
            --cuda_home $CUDA_HOME --cudnn_home $CUDNN_HOME --build_wheel --skip_tests \
            --use_tensorrt --tensorrt_home $TRT_HOME \
            --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF \
            --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=80

pythonVersion=`python -c "import sys;t='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t)";`
echo "python version ($pythonVersion)"
if [ "${pythonVersion}" = "3.10" ]; then
  pip install build/Linux/Release/dist/onnxruntime_gpu-1.17.0-cp310-cp310-linux_x86_64.whl protobuf==3.20.2 --force-reinstall
fi