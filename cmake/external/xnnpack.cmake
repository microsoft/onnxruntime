set(XNNPACK_DIR external/XNNPACK)
set(XNNPACK_INCLUDE_DIR ${XNNPACK_DIR}/include)
set(XNNPACK_USE_SYSTEM_LIBS ON CACHE INTERNAL "")
set(XNNPACK_BUILD_TESTS OFF CACHE INTERNAL "")
set(XNNPACK_BUILD_BENCHMARKS OFF CACHE INTERNAL "")
set(FP16_BUILD_TESTS OFF CACHE INTERNAL "")
set(FP16_BUILD_BENCHMARKS OFF CACHE INTERNAL "")
set(CLOG_SOURCE_DIR "${PYTORCH_CPUINFO_DIR}/deps/clog")
set(CPUINFO_SOURCE_DIR ${PYTORCH_CPUINFO_DIR})

# BF16 instructions cause ICE in Android NDK compiler
if(CMAKE_ANDROID_ARCH_ABI STREQUAL armeabi-v7a)
  set(XNNPACK_ENABLE_ARM_BF16 OFF)
ENDIF()

if(onnxruntime_BUILD_WEBASSEMBLY OR CMAKE_SYSTEM_NAME STREQUAL "iOS")
  execute_process(COMMAND git apply --ignore-space-change --ignore-whitespace ${PROJECT_SOURCE_DIR}/patches/xnnpack/AddEmscriptenAndIosSupport.patch WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/${XNNPACK_DIR})
endif()

add_subdirectory(external/FP16)
add_subdirectory(external/pthreadpool)
add_subdirectory(external/XNNPACK)

set_target_properties(fp16 PROPERTIES FOLDER "External/Xnnpack")
set_target_properties(pthreadpool PROPERTIES FOLDER "External/Xnnpack")
set_target_properties(XNNPACK PROPERTIES FOLDER "External/Xnnpack")

set(onnxruntime_EXTERNAL_LIBRARIES_XNNPACK XNNPACK pthreadpool)
list(APPEND onnxruntime_EXTERNAL_LIBRARIES ${onnxruntime_EXTERNAL_LIBRARIES_XNNPACK})

# the XNNPACK CMake setup doesn't include the WASM kernels so we have to manually set those up
if(onnxruntime_BUILD_WEBASSEMBLY)
  file(READ "${XNNPACK_DIR}/BUILD.bazel" xnnpack_bazel_config)

  # Replace newlines with semicolon so that it is treated as a list by CMake
  # Also replace '[' and ']' so the bazel source lists don't get parsed as a nested list by cmake
  string(REPLACE "\n" ";" xnnpack_bazel_config "${xnnpack_bazel_config}")
  string(REPLACE "[" "{" xnnpack_bazel_config "${xnnpack_bazel_config}")
  string(REPLACE "]" "}" xnnpack_bazel_config "${xnnpack_bazel_config}")

  function(GetSrcListFromBazel src_list_name target_srcs)
    set(_InSection FALSE)
    set(bazel_srcs "")

    foreach(_line ${xnnpack_bazel_config})
      if(NOT _InSection)
        if(_line MATCHES "^${src_list_name} = \\{")
          set(_InSection TRUE)
        endif()
      else()
        if(_line MATCHES "^\\}")
          set(_InSection FALSE)
        else()
          # parse filename from quoted string with trailing comma
          string(REPLACE "\"" "" _line "${_line}")
          string(REPLACE "," "" _line "${_line}")
          string(STRIP "${_line}" _line)

          list(APPEND bazel_srcs "${XNNPACK_DIR}/${_line}")
        endif()
      endif()
    endforeach()

    set(${target_srcs} ${bazel_srcs} PARENT_SCOPE)
  endfunction()

  GetSrcListFromBazel("PROD_SCALAR_WASM_MICROKERNEL_SRCS" prod_scalar_wasm_srcs)
  GetSrcListFromBazel("ALL_WASM_MICROKERNEL_SRCS" all_wasm_srcs)
  GetSrcListFromBazel("WASM32_ASM_MICROKERNEL_SRCS" wasm32_asm_srcs)

  message(DEBUG "prod_scalar_wasm_srcs: ${prod_scalar_wasm_srcs}\n")
  message(DEBUG "all_wasm_srcs: ${all_wasm_srcs}\n")
  message(DEBUG "wasm32_asm_srcs: ${wasm32_asm_srcs}\n")

  message("Adding WebAssembly Source Files to XNNPACK")
  set(wasm_srcs "")
  list(APPEND wasm_srcs ${prod_scalar_wasm_srcs})
  list(APPEND wasm_srcs ${all_wasm_srcs})
  list(APPEND wasm_srcs ${wasm32_asm_srcs})

  target_sources(XNNPACK PRIVATE ${wasm_srcs})

  if(onnxruntime_ENABLE_WEBASSEMBLY_SIMD)
    GetSrcListFromBazel("ALL_WASMSIMD_MICROKERNEL_SRCS" all_wasmsimd_srcs)
    message(DEBUG "all_wasmsimd_srcs: ${all_wasmsimd_srcs}")
    target_sources(XNNPACK PRIVATE ${all_wasmsimd_srcs})
  endif()
endif()
