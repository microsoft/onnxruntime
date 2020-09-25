export CUDACXX="/usr/local/cuda-10.1/bin/nvcc"
rm -rf build/Linux/RelWithDebInfo/dist/
pip uninstall onnxruntime_gpu
./build.sh --enable_training --use_cuda --config=Debug --cuda_version=10.1 --cuda_home=/usr/local/cuda-10.1 --cudnn_home=/usr/local/cuda-10.1 --update --build --build_wheel
pip install  build/Linux/Debug/dist/onnxruntime_gpu-1.4.0-cp36-cp36m-linux_x86_64.whl

#./build.sh --enable_training --use_cuda --config=RelWithDebInfo --cuda_version=10.1 --cuda_home=/usr/local/cuda-10.1 --cudnn_home=/usr/local/cuda-10.1 --update --build --build_wheel --cmake_extra_defines onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS=1
#pip install  build/Linux/RelWithDebInfo/dist/onnxruntime_gpu-1.4.0-cp36-cp36m-linux_x86_64.whl


#
