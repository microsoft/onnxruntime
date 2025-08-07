# Precompiled header configuration for onnxruntime_test_all
target_include_directories(onnxruntime_test_all PRIVATE
  "Q:/src/onnxruntime/onnxruntime/test"
  "Q:/src/onnxruntime/build/Windows/RelWithDebInfo/_deps/googletest-src/googletest/include"
  "Q:/src/onnxruntime/include/onnxruntime"
  "Q:/src/onnxruntime/build/Windows/RelWithDebInfo/_deps/onnx-src"
  "Q:/src/onnxruntime/build/Windows/RelWithDebInfo/_deps/googletest-src/googletest/include/gtest/internal"
)
if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  # Visual Studio PCH
  target_precompile_headers(onnxruntime_test_all PRIVATE
    "Q:/src/onnxruntime/onnxruntime/cmake/test_pch.h"
  )
  endif()

# Exclude certain files that might conflict with PCH
set(PCH_EXCLUDE_FILES
  # Add any problematic source files here
  Q:/src/onnxruntime/onnxruntime/test/framework/tensor_shape_test.cc
)

foreach(file ${PCH_EXCLUDE_FILES})
  set_source_files_properties(${file} PROPERTIES
    SKIP_PRECOMPILE_HEADERS ON
  )
endforeach()
