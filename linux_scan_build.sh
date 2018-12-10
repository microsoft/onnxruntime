export PATH=/opt/protobuf36/bin:$PATH 
export CC=clang
export CXX=clang++ 
rm -rf scan_build_dir
mkdir scan_build_dir
(cd scan_build_dir && scan-build cmake -DCMAKE_BUILD_TYPE=Debug -Donnxruntime_ENABLE_PYTHON=ON -DPYTHON_EXECUTABLE=/usr/bin/python3 -Donnxruntime_USE_PREBUILT_PB=ON -DONNX_CUSTOM_PROTOC_EXECUTABLE=/opt/protobuf36/bin/protoc -Deigen_SOURCE_PATH=/usr/include/eigen3 -Donnxruntime_USE_PREINSTALLED_EIGEN=ON $HOME/src/Lotus/cmake && scan-build -stats make -j$(nproc))
