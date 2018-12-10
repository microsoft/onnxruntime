export PATH=/usr/local/cuda-9.2/bin:$PATH
cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug -Dlotus_ENABLE_PYTHON=ON -DPYTHON_EXECUTABLE=/usr/bin/python3 -Dlotus_USE_PREBUILT_PB=ON -DONNX_CUSTOM_PROTOC_EXECUTABLE=/usr/bin/protoc -Deigen_SOURCE_PATH=/usr/include/eigen3 -Dlotus_USE_PREINSTALLED_EIGEN=ON -Dlotus_USE_CUDA=ON -Dlotus_CUDNN_HOME=/usr/local/cuda $HOME/src/Lotus/cmake
