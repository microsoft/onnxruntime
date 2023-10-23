set(XNNPACK_USE_SYSTEM_LIBS ON CACHE INTERNAL "")
set(XNNPACK_BUILD_TESTS OFF CACHE INTERNAL "")
set(XNNPACK_BUILD_BENCHMARKS OFF CACHE INTERNAL "")
set(FP16_BUILD_TESTS OFF CACHE INTERNAL "")
set(FP16_BUILD_BENCHMARKS OFF CACHE INTERNAL "")
set(PTHREADPOOL_BUILD_TESTS OFF CACHE INTERNAL "")
set(PTHREADPOOL_BUILD_BENCHMARKS OFF CACHE INTERNAL "")

# BF16 instructions cause ICE in Android NDK compiler
if(CMAKE_ANDROID_ARCH_ABI STREQUAL armeabi-v7a)
  set(XNNPACK_ENABLE_ARM_BF16 OFF)
ENDIF()

# fp16 depends on psimd
FetchContent_Declare(psimd URL ${DEP_URL_psimd} URL_HASH SHA1=${DEP_SHA1_psimd})
onnxruntime_fetchcontent_makeavailable(psimd)
set(PSIMD_SOURCE_DIR ${psimd_SOURCE_DIR})
FetchContent_Declare(fp16 URL ${DEP_URL_fp16} URL_HASH SHA1=${DEP_SHA1_fp16})
onnxruntime_fetchcontent_makeavailable(fp16)

# pthreadpool depends on fxdiv
FetchContent_Declare(fxdiv URL ${DEP_URL_fxdiv} URL_HASH SHA1=${DEP_SHA1_fxdiv})
onnxruntime_fetchcontent_makeavailable(fxdiv)
set(FXDIV_SOURCE_DIR ${fxdiv_SOURCE_DIR})

FetchContent_Declare(pthreadpool URL ${DEP_URL_pthreadpool} URL_HASH SHA1=${DEP_SHA1_pthreadpool})
onnxruntime_fetchcontent_makeavailable(pthreadpool)

FetchContent_Declare(googlexnnpack URL ${DEP_URL_googlexnnpack} URL_HASH SHA1=${DEP_SHA1_googlexnnpack}
                     PATCH_COMMAND ${Patch_EXECUTABLE} --binary --ignore-whitespace -p1 < ${PROJECT_SOURCE_DIR}/patches/xnnpack/AddEmscriptenAndIosSupport.patch
                    )
onnxruntime_fetchcontent_makeavailable(googlexnnpack)
set(XNNPACK_DIR ${googlexnnpack_SOURCE_DIR})
set(XNNPACK_INCLUDE_DIR ${XNNPACK_DIR}/include)

set(onnxruntime_EXTERNAL_LIBRARIES_XNNPACK XNNPACK pthreadpool)


# the XNNPACK CMake setup doesn't include the WASM kernels so we have to manually set those up
if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  # The source lists in BUILD.bazel are defined like this.
  # wasm_srcs = ["src/amalgam/gen/wasm.c"],
  #  wasmrelaxedsimd_srcs = [
  #  "src/amalgam/gen/wasm.c",
  #  "src/amalgam/gen/wasmrelaxedsimd.c",
  #  "src/amalgam/gen/wasmsimd.c",
  # ],
  # wasmsimd_srcs = [
  #  "src/amalgam/gen/wasm.c",
  #  "src/amalgam/gen/wasmsimd.c",
  # ],

#   xnnpack_cc_library(
#     name = "wasm_prod_microkernels",
#     gcc_copts = xnnpack_gcc_std_copts() + [
#         "-fno-fast-math",
#         "-fno-math-errno",
#     ],
#     msvc_copts = xnnpack_msvc_std_copts(),
#     wasm_srcs = ["src/amalgam/gen/wasm.c"],
#     wasmrelaxedsimd_srcs = [
#         "src/amalgam/gen/wasm.c",
#         "src/amalgam/gen/wasmrelaxedsimd.c",
#         "src/amalgam/gen/wasmsimd.c",
#     ],
#     wasmsimd_srcs = [
#         "src/amalgam/gen/wasm.c",
#         "src/amalgam/gen/wasmsimd.c",
#     ],
#     deps = [
#         ":common",
#         ":math",
#         ":microkernels_h",
#         ":microparams",
#         ":prefetch",
#         ":tables",
#         ":unaligned",
#     ],
#   )

  message("Adding WebAssembly Source Files to XNNPACK")
  set(wasm_srcs "")
  # :common
  list(APPEND wasm_srcs ${XNNPACK_DIR}/src/xnnpack/common.h)
  # :math
  list(APPEND wasm_srcs ${XNNPACK_DIR}/src/xnnpack/math.h)
  # :microkernels_h
  list(APPEND wasm_srcs ${XNNPACK_DIR}/src/xnnpack/cache.h
                        ${XNNPACK_DIR}/src/xnnpack/intrinsics-polyfill.h
                        ${XNNPACK_DIR}/src/xnnpack/math-stubs.h
                        ${XNNPACK_DIR}/src/xnnpack/requantization-stubs.h)
  # :microparams
  list(APPEND wasm_srcs ${XNNPACK_DIR}/src/xnnpack/microparams.h)
  # :prefetch
  list(APPEND wasm_srcs ${XNNPACK_DIR}/src/xnnpack/prefetch.h)
  # :tables
  list(APPEND wasm_srcs ${XNNPACK_DIR}/src/tables/exp2-k-over-64.c
                        ${XNNPACK_DIR}/src/tables/exp2-k-over-2048.c
                        ${XNNPACK_DIR}/src/tables/exp2minus-k-over-4.c
                        ${XNNPACK_DIR}/src/tables/exp2minus-k-over-8.c
                        ${XNNPACK_DIR}/src/tables/exp2minus-k-over-16.c
                        ${XNNPACK_DIR}/src/tables/exp2minus-k-over-32.c
                        ${XNNPACK_DIR}/src/tables/exp2minus-k-over-64.c
                        ${XNNPACK_DIR}/src/tables/exp2minus-k-over-2048.c
                        ${XNNPACK_DIR}/src/tables/vlog.c)
  # :unaligned
  list(APPEND wasm_srcs ${XNNPACK_DIR}/src/xnnpack/unaligned.h)

  # not explicitly references in BAZEL.build but unresolved symbols if missing.
  # e.g. see src/configs/abgpool-config.c has this which is implemented in scalar.c
  # #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  #   qu8_avgpool_config.unipass = (xnn_avgpool_unipass_ukernel_fn) xnn_qu8_avgpool_minmax_fp32_ukernel_9x__scalar_imagic_c1;
  list(APPEND wasm_srcs ${XNNPACK_DIR}/src/amalgam/gen/scalar.c)

  list(APPEND wasm_srcs ${XNNPACK_DIR}/src/amalgam/gen/wasm.c)

  if(onnxruntime_ENABLE_WEBASSEMBLY_SIMD)
    #target_compile_definitions(XNNPACK PRIVATE XNN_ARCH_WASMSIMD)
    #target_compile_definitions(XNNPACK PRIVATE __wasm_simd128__)
    list(APPEND wasm_srcs ${XNNPACK_DIR}/src/amalgam/gen/wasmsimd.c)
    target_compile_options(XNNPACK PRIVATE "-msimd128")
  endif()

  target_sources(XNNPACK PRIVATE ${wasm_srcs})

  # add flags from BAZEL.build
  target_compile_options(XNNPACK PRIVATE "-fno-fast-math")
  target_compile_options(XNNPACK PRIVATE "-fno-math-errno")
endif()
