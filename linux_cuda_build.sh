export PATH=/usr/local/cuda-9.2/bin:$PATH
cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug -Donnxruntime_ENABLE_PYTHON=ON -DPYTHON_EXECUTABLE=/usr/bin/python3 -Donnxruntime_USE_PREBUILT_PB=ON -DONNX_CUSTOM_PROTOC_EXECUTABLE=/usr/bin/protoc -Deigen_SOURCE_PATH=/usr/include/eigen3 -Donnxruntime_USE_PREINSTALLED_EIGEN=ON -Donnxruntime_USE_CUDA=ON -Donnxruntime_CUDNN_HOME=/usr/local/cuda $HOME/src/Lotus/cmake
