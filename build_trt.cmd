IF {%VCToolsVersion%}=={} call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" amd64

set CudaToolkitDir=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2

.\build.bat --config Release --build_dir .\build --build_shared_lib --parallel --build_wheel ^
            --cmake_generator "Visual Studio 17 2022" ^
            --use_cuda --cuda_version 12.2 --cuda_home "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2" ^
            --use_tensorrt --tensorrt_home "D:\nvidia\TensorRT-8.6.1.6.Windows10.x86_64.cuda-12.0\TensorRT-8.6.1.6" ^
            --cudnn_home d:\nvidia\cudnn-windows-x86_64-8.9.4.25_cuda12-archive ^
            --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=89 ^
            --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF  ^
            --skip_tests
