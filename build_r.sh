export CUDA_HOME=/usr/local/cuda-12.2
export CUDNN_HOME=/usr/lib/x86_64-linux-gnu/
export CUDACXX=/usr/local/cuda-12.2/bin/nvcc

cd /work/tlwu/git/onnxruntime

sh build.sh --config Release  --build_shared_lib --parallel  --use_cuda --cuda_version 12.2 \
            --cuda_home $CUDA_HOME --cudnn_home $CUDNN_HOME --build_wheel --skip_tests \
            --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=80
