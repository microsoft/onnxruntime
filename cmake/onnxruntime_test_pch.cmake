# Precompiled header configuration for onnxruntime_test_all

if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  # Visual Studio PCH
  target_precompile_headers(onnxruntime_test_all PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/test_pch.h"
  )
  target_precompile_headers(onnxruntime_provider_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/test_pch.h"
  )
endif()

# Exclude certain files that might conflict with PCH
set(PCH_EXCLUDE_FILES
  # Add any problematic source files here
  "${TEST_SRC_DIR}/framework/tensor_shape_test.cc"
)

foreach(file ${PCH_EXCLUDE_FILES})
  set_source_files_properties(${file} PROPERTIES
    SKIP_PRECOMPILE_HEADERS ON
  )
endforeach()
