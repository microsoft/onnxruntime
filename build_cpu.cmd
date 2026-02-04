IF "%VCToolsInstallDir%"=="" call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

build.bat --cmake_generator "Visual Studio 17 2022" --config Release --build_dir build\cpu --build_wheel ^
          --parallel 4 --nvcc_threads 1 --build_shared_lib ^
          --skip_tests ^
          --use_binskim_compliant_compile_flags ^
          --cmake_extra_defines "onnxruntime_BUILD_UNIT_TESTS=OFF" ^
          --cmake_extra_defines "FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER"