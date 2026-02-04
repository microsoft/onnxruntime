IF "%VCToolsInstallDir%"=="" call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

build.bat --cmake_generator "Visual Studio 17 2022" --config Release --build_dir build\cuda130 --build_wheel ^
          --parallel 4 --nvcc_threads 1 --build_shared_lib ^
          --use_cuda --cuda_version "13.0" --cuda_home "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0" ^
          --cudnn_home "D:\cudnn\9.13.0.50_cuda13" ^
          --cmake_extra_defines "CMAKE_CUDA_ARCHITECTURES=native" ^
          --build_nuget ^
          --skip_tests ^
          --use_binskim_compliant_compile_flags ^
          --cmake_extra_defines "onnxruntime_BUILD_UNIT_TESTS=ON" ^
          --cmake_extra_defines "onnxruntime_ENABLE_NVTX_PROFILE=ON" ^
          --cmake_extra_defines "onnxruntime_ENABLE_CUDA_LINE_NUMBER_INFO=ON" ^
          --cmake_extra_defines "FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER"

REM --use_vcpkg
REM onnxruntime_ENABLE_NVTX_PROFILE=ON --enable_cuda_line_info
