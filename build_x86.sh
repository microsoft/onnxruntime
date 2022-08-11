if [ x"$1" == x"g" ];then
    conf="Debug"
else
    conf="Release"
fi
#bash build.sh  --cmake_extra_defines CMAKE_EXPORT_COMPILE_COMMANDS=ON  --use_cuda --cudnn_home $cuDNN_PATH   --cuda_home $CUDA_PATH  --config=Debug --paralle 20
bash build.sh --cmake_generator "Ninja"  --cmake_extra_defines CMAKE_EXPORT_COMPILE_COMMANDS=ON --skip_tests  --build_wheel --use_xnnpack --paralle 20
