IF "VCToolsInstallDir"=="" call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64.bat"

build.bat --cmake_generator "Visual Studio 16 2019" --config Release --build_dir build --build_wheel ^
          --parallel 4 --nvcc_threads 1 --build_shared_lib ^
          --use_cuda --cuda_version "11.8" --cuda_home "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8" ^
          --cudnn_home "D:\cudnn\8.9.7.29_cuda11" ^
          --cmake_extra_defines "CMAKE_CUDA_ARCHITECTURES=61;86" ^
          --use_binskim_compliant_compile_flags ^
          --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF --skip_tests

REM onnxruntime_ENABLE_NVTX_PROFILE=ON --enable_cuda_line_info
