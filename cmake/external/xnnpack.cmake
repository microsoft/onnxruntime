set(XNNPACK_DIR external/XNNPACK)
set(XNNPACK_INCLUDE_DIR ${XNNPACK_DIR}/include)
set(XNNPACK_USE_SYSTEM_LIBS ON CACHE INTERNAL "")
set(XNNPACK_BUILD_TESTS OFF CACHE INTERNAL "")
set(XNNPACK_BUILD_BENCHMARKS OFF CACHE INTERNAL "")
set(FP16_BUILD_TESTS OFF CACHE INTERNAL "")
set(FP16_BUILD_BENCHMARKS OFF CACHE INTERNAL "")
set(CLOG_SOURCE_DIR "${PYTORCH_CPUINFO_DIR}/deps/clog")
set(CPUINFO_SOURCE_DIR ${PYTORCH_CPUINFO_DIR})

if (onnxruntime_BUILD_WEBASSEMBLY)
  execute_process(COMMAND  git apply --ignore-space-change --ignore-whitespace ${PROJECT_SOURCE_DIR}/patches/xnnpack/AddEmscriptenSupport.patch WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/${XNNPACK_DIR})
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
if (onnxruntime_BUILD_WEBASSEMBLY)
  if (onnxruntime_ENABLE_WEBASSEMBLY_THREADS)
    target_compile_options(XNNPACK PRIVATE "-pthread")
  endif()

  message("Adding WebAssembly Source Files to XNNPACK")
  set(wasm_src_patterns "${XNNPACK_DIR}/src/wasm-*.c"
                        "${XNNPACK_DIR}/src/*-wasm-*.c"
                        "${XNNPACK_DIR}/src/*-wasm.c")
  set(wasm32_asm_src_patterns "${XNNPACK_DIR}/src/wasm_shr_*.S")

  file(GLOB_RECURSE XNNPACK_WASM_MICROKERNEL_SRCS CONFIGURE_DEPENDS ${wasm_src_patterns})
  file(GLOB_RECURSE XNNPACK_WASM32_ASM_MICROKERNEL_SRCS CONFIGURE_DEPENDS ${wasm32_asm_src_patterns})

  message(DEBUG "XNNPACK_WASM_MICROKERNEL_SRCS:${XNNPACK_WASM_MICROKERNEL_SRCS}")
  message(DEBUG "XNNPACK_WASM32_ASM_MICROKERNEL_SRCS:${XNNPACK_WASM32_ASM_MICROKERNEL_SRCS}")

  target_sources(XNNPACK PRIVATE ${XNNPACK_WASM_MICROKERNEL_SRCS}
                                 ${XNNPACK_WASM32_ASM_MICROKERNEL_SRCS})

  if (onnxruntime_ENABLE_WEBASSEMBLY_SIMD)
    target_compile_options(XNNPACK PRIVATE "-msimd128")

    set(wasmsimd_src_patterns "${XNNPACK_DIR}/src/wasmsimd-*.c"
                              "${XNNPACK_DIR}/src/*-wasmsimd-*.c"
                              "${XNNPACK_DIR}/src/*-wasmsimd.c"
                              "${XNNPACK_DIR}/src/*/wasmsimd.c")

    file(GLOB_RECURSE XNNPACK_WASMSIMD_MICROKERNEL_SRCS CONFIGURE_DEPENDS ${wasmsimd_src_patterns})
    message(DEBUG "XNNPACK_WASMSIMD_MICROKERNEL_SRCS:${XNNPACK_WASMSIMD_MICROKERNEL_SRCS}")

    target_sources(XNNPACK PRIVATE ${XNNPACK_WASMSIMD_MICROKERNEL_SRCS})
  endif()
endif()