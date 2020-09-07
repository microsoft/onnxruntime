rm -rf build/Linux/RelWithDebInfo/dist/
pip uninstall onnxruntime_gpu
./build.sh --enable_training --use_cuda --config=RelWithDebInfo --cuda_home=/usr/local/cuda --cudnn_home=/usr/local/cuda --parallel --build_wheel  --mpi_home=/opt/ompi4/ --update --build 

pip install  build/Linux/RelWithDebInfo/dist/onnxruntime_gpu-1.4.0-cp36-cp36m-linux_x86_64.whl

#--cmake_extra_defines onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS=1
