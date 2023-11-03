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

  GetSrcListFromBazel("OPERATOR_SRCS" operator_srcs)
  GetSrcListFromBazel("TABLE_SRCS" table_srcs)
  list(APPEND wasm_srcs ${operator_srcs} ${table_srcs})

  # kernels
  list(APPEND wasm_srcs ${XNNPACK_DIR}/src/amalgam/gen/scalar.c)
  list(APPEND wasm_srcs ${XNNPACK_DIR}/src/amalgam/gen/wasm.c)

  if(onnxruntime_ENABLE_WEBASSEMBLY_SIMD)
    list(APPEND wasm_srcs ${XNNPACK_DIR}/src/amalgam/gen/wasmsimd.c)
    target_compile_options(XNNPACK PRIVATE "-msimd128")
  endif()

  message(DEBUG "wasm_srcs: ${wasm_srcs}\n")
  target_sources(XNNPACK PRIVATE ${wasm_srcs})

  # add flags from BAZEL.build
  target_compile_options(XNNPACK PRIVATE "-fno-fast-math")
  target_compile_options(XNNPACK PRIVATE "-fno-math-errno")
endif()
