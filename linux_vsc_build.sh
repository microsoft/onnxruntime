export PATH=/opt/protobuf36/bin:$PATH 
which eclipse
CC=clang CXX=clang++ cmake -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Debug -Donnxruntime_ENABLE_PYTHON=ON -DPYTHON_EXECUTABLE=/usr/bin/python3 -Donnxruntime_USE_PREBUILT_PB=ON -DONNX_CUSTOM_PROTOC_EXECUTABLE=/opt/protobuf36/bin/protoc -Deigen_SOURCE_PATH=/home/chasun/os/eigen/3.3.5 -Donnxruntime_USE_PREINSTALLED_EIGEN=ON $HOME/src/Lotus/cmake
