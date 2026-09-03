# Precompiled header configuration for the CPU providers target

if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  # Visual Studio PCH
  target_precompile_headers(${onnxruntime_providers_target} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/providers_pch.h"
  )
endif()
