rm -rf build
# ./build.sh --build_dir build --config RelWithDebInfo --parallel --use_hip --hip_home /opt/rocm --enable_training
./build.sh --build_dir build --config Release --use_migraphx --migraphx_home  /opt/rocm 