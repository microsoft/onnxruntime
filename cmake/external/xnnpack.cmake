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
  # See source lists in _deps/googlexnnpack-src/BUILD.bazel for wasm_prod_microkernels
  message("Adding WebAssembly Source Files to XNNPACK")
  set(wasm_srcs "")
  list(APPEND wasm_srcs ${XNNPACK_DIR}/src/xnnpack/common.h)
  list(APPEND wasm_srcs ${XNNPACK_DIR}/src/xnnpack/math.h)
  list(APPEND wasm_srcs ${XNNPACK_DIR}/src/xnnpack/microparams.h)
  list(APPEND wasm_srcs ${XNNPACK_DIR}/src/xnnpack/prefetch.h)
  list(APPEND wasm_srcs ${XNNPACK_DIR}/src/xnnpack/unaligned.h)

  # :microkernels_h
  list(APPEND wasm_srcs ${XNNPACK_DIR}/src/xnnpack/cache.h
                        ${XNNPACK_DIR}/src/xnnpack/intrinsics-polyfill.h
                        ${XNNPACK_DIR}/src/xnnpack/math-stubs.h
                        ${XNNPACK_DIR}/src/xnnpack/requantization-stubs.h)
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
  # operators
  list(APPEND wasm_srcs ${XNNPACK_DIR}/src/xnnpack/compute.h
                        ${XNNPACK_DIR}/src/xnnpack/operator.h
                        ${XNNPACK_DIR}/src/operator-delete.c
                        ${XNNPACK_DIR}/src/operator-run.c
                        ${XNNPACK_DIR}/src/operators/argmax-pooling-nhwc.c
                        ${XNNPACK_DIR}/src/operators/average-pooling-nhwc.c
                        ${XNNPACK_DIR}/src/operators/batch-matrix-multiply-nc.c
                        ${XNNPACK_DIR}/src/operators/binary-elementwise-nd.c
                        ${XNNPACK_DIR}/src/operators/channel-shuffle-nc.c
                        ${XNNPACK_DIR}/src/operators/constant-pad-nd.c
                        ${XNNPACK_DIR}/src/operators/convolution-nchw.c
                        ${XNNPACK_DIR}/src/operators/convolution-nhwc.c
                        ${XNNPACK_DIR}/src/operators/deconvolution-nhwc.c
                        ${XNNPACK_DIR}/src/operators/dynamic-fully-connected-nc.c
                        ${XNNPACK_DIR}/src/operators/fully-connected-nc.c
                        ${XNNPACK_DIR}/src/operators/global-average-pooling-ncw.c
                        ${XNNPACK_DIR}/src/operators/global-average-pooling-nwc.c
                        ${XNNPACK_DIR}/src/operators/lut-elementwise-nc.c
                        ${XNNPACK_DIR}/src/operators/max-pooling-nhwc.c
                        ${XNNPACK_DIR}/src/operators/prelu-nc.c
                        ${XNNPACK_DIR}/src/operators/reduce-nd.c
                        ${XNNPACK_DIR}/src/operators/resize-bilinear-nchw.c
                        ${XNNPACK_DIR}/src/operators/resize-bilinear-nhwc.c
                        ${XNNPACK_DIR}/src/operators/rope-nthc.c
                        ${XNNPACK_DIR}/src/operators/scaled-dot-product-attention-nhtc.c
                        ${XNNPACK_DIR}/src/operators/slice-nd.c
                        ${XNNPACK_DIR}/src/operators/softmax-nc.c
                        ${XNNPACK_DIR}/src/operators/transpose-nd.c
                        ${XNNPACK_DIR}/src/operators/unary-elementwise-nc.c
                        ${XNNPACK_DIR}/src/operators/unpooling-nhwc.c
  )

  # kernels
  list(APPEND wasm_srcs ${XNNPACK_DIR}/src/amalgam/gen/scalar.c)
  list(APPEND wasm_srcs ${XNNPACK_DIR}/src/amalgam/gen/wasm.c)

  if(onnxruntime_ENABLE_WEBASSEMBLY_SIMD)
    list(APPEND wasm_srcs ${XNNPACK_DIR}/src/amalgam/gen/wasmsimd.c)
    target_compile_options(XNNPACK PRIVATE "-msimd128")
  endif()

  target_sources(XNNPACK PRIVATE ${wasm_srcs})

  # add flags from BAZEL.build
  target_compile_options(XNNPACK PRIVATE "-fno-fast-math")
  target_compile_options(XNNPACK PRIVATE "-fno-math-errno")
endif()
