# Precompiled header configuration for onnxruntime_providers
target_include_directories(onnxruntime_providers PRIVATE
  "Q:/src/onnxruntime/include/onnxruntime"
  "Q:/src/onnxruntime/build/Windows/RelWithDebInfo/_deps/onnx-src"
  "Q:/src/onnxruntime/onnxruntime"
)

if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  # Visual Studio PCH
  target_precompile_headers(onnxruntime_providers PRIVATE
    "Q:/src/onnxruntime/cmake/providers_pch.h"
  )
endif()

# Exclude certain files that might conflict with PCH
set(PROVIDERS_PCH_EXCLUDE_FILES
  # Add any problematic source files here if needed
  # Example: Q:/src/onnxruntime/onnxruntime/core/providers/cpu/some_problematic_file.cc
)

foreach(file ${PROVIDERS_PCH_EXCLUDE_FILES})
  set_source_files_properties(${file} PROPERTIES
    SKIP_PRECOMPILE_HEADERS ON
  )
endforeach()
