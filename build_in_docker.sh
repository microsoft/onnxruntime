export CUDA_HOME=/usr/local/cuda-11.1
export CUDNN_HOME=/usr/local/cuda-11.1
export CUDACXX=$CUDA_HOME/bin/nvcc

pip uninstall onnxruntime_gpu --yes

ORT_PATH=/bert_ort/pengwa/dev4/onnxruntime

rm -rf ${ORT_PATH}/build/Linux/RelWithDebInfo/dist/onnxruntime_gpu-1.7.0-cp37-cp37m-linux_x86_64.whl

./build.sh --config RelWithDebInfo --use_cuda --enable_training --build_wheel --skip_tests --parallel 8 --enable_language_interop_ops


pip install ${ORT_PATH}/build/Linux/RelWithDebInfo/dist/onnxruntime_gpu-1.7.0-cp37-cp37m-linux_x86_64.whl
