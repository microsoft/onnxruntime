IF "%VCToolsInstallDir%"=="" call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

build.bat --cmake_generator "Visual Studio 17 2022" --config Release --build_dir build\cuda129 --build_wheel ^
          --parallel 4 --nvcc_threads 1 --build_shared_lib ^
          --use_cuda --cuda_version "12.9" --cuda_home "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9" ^
          --cudnn_home "D:\cudnn\9.13.0.50_cuda12" ^
          --build_nuget ^
          --skip_tests ^
          --use_binskim_compliant_compile_flags ^
          --cmake_extra_defines "CMAKE_CUDA_ARCHITECTURES=120" ^
          --cmake_extra_defines "onnxruntime_USE_FPA_INTB_GEMM=ON" ^
          --cmake_extra_defines "onnxruntime_BUILD_UNIT_TESTS=OFF" ^
          --cmake_extra_defines "FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER"


REM onnxruntime_ENABLE_NVTX_PROFILE=ON --enable_cuda_line_info
