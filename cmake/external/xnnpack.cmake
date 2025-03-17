set(XNNPACK_USE_SYSTEM_LIBS ON CACHE INTERNAL "")
set(XNNPACK_BUILD_TESTS OFF CACHE INTERNAL "")
set(XNNPACK_BUILD_BENCHMARKS OFF CACHE INTERNAL "")

set(PTHREADPOOL_BUILD_TESTS OFF CACHE INTERNAL "")
set(PTHREADPOOL_BUILD_BENCHMARKS OFF CACHE INTERNAL "")
set(KLEIDIAI_BUILD_TESTS OFF CACHE INTERNAL "")
set(KLEIDIAI_BUILD_BENCHMARK OFF CACHE INTERNAL "")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^riscv64.*")
  set(XNNPACK_USE_SYSTEM_LIBS OFF)
endif()

# BF16 instructions cause ICE in Android NDK compiler
if(CMAKE_ANDROID_ARCH_ABI STREQUAL armeabi-v7a)
  set(XNNPACK_ENABLE_ARM_BF16 OFF)
endif()

# pthreadpool depends on fxdiv
FetchContent_Declare(fxdiv URL ${DEP_URL_fxdiv} URL_HASH SHA1=${DEP_SHA1_fxdiv})
onnxruntime_fetchcontent_makeavailable(fxdiv)
set(FXDIV_SOURCE_DIR ${fxdiv_SOURCE_DIR})

FetchContent_Declare(pthreadpool URL ${DEP_URL_pthreadpool} URL_HASH SHA1=${DEP_SHA1_pthreadpool})
onnxruntime_fetchcontent_makeavailable(pthreadpool)

# ---  Determine target processor
# Why ORT_TARGET_PROCESSOR is only for XNNPACK
# So far, only Onnxruntime + XNNPack only allow one target processor.
# And we support Mac universal package, so,
# CMAKE_OSX_ARCHITECTURES_COUNT greater than 1 is allowed in other places.
IF(CMAKE_OSX_ARCHITECTURES)
  LIST(LENGTH CMAKE_OSX_ARCHITECTURES CMAKE_OSX_ARCHITECTURES_COUNT)
  IF(CMAKE_OSX_ARCHITECTURES_COUNT GREATER 1)
    MESSAGE(STATUS "Building ONNX Runtime with XNNPACK and multiple OSX architectures is not supported. Got:(${CMAKE_OSX_ARCHITECTURES}). "
      "Please specify a single architecture in CMAKE_OSX_ARCHITECTURES and re-configure. ")
  ENDIF()
  IF(NOT CMAKE_OSX_ARCHITECTURES MATCHES "^(x86_64|arm64|arm64e|arm64_32)$")
    MESSAGE(FATAL_ERROR "Unrecognized CMAKE_OSX_ARCHITECTURES value \"${CMAKE_OSX_ARCHITECTURES}\"")
  ENDIF()
  SET(ORT_TARGET_PROCESSOR "${CMAKE_OSX_ARCHITECTURES}")
  ADD_COMPILE_OPTIONS("-Wno-shorten-64-to-32")
ELSEIF(CMAKE_GENERATOR MATCHES "^Visual Studio " AND CMAKE_GENERATOR_PLATFORM)
  IF(CMAKE_GENERATOR_PLATFORM MATCHES "^Win32")
    SET(ORT_TARGET_PROCESSOR "x86")
  ELSEIF(CMAKE_GENERATOR_PLATFORM MATCHES "^x64")
    SET(ORT_TARGET_PROCESSOR "x86_64")
  ELSEIF(CMAKE_GENERATOR_PLATFORM MATCHES "^ARM64")
    SET(ORT_TARGET_PROCESSOR "arm64")
  ELSEIF(CMAKE_GENERATOR_PLATFORM MATCHES "^ARM64EC")
    SET(ORT_TARGET_PROCESSOR "arm64")
  ELSE()
    MESSAGE(FATAL_ERROR "Unsupported Visual Studio architecture \"${CMAKE_GENERATOR_PLATFORM}\"")
  ENDIF()
ELSEIF(CMAKE_SYSTEM_PROCESSOR MATCHES "^i[3-7]86$")
  SET(ORT_TARGET_PROCESSOR "x86")
ELSEIF(CMAKE_SYSTEM_PROCESSOR STREQUAL "AMD64")
  SET(ORT_TARGET_PROCESSOR "x86_64")
ELSEIF(CMAKE_SYSTEM_PROCESSOR MATCHES "^armv[5-8]")
  SET(ORT_TARGET_PROCESSOR "arm")
ELSEIF(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
  SET(ORT_TARGET_PROCESSOR "arm64")
ELSEIF(CMAKE_SYSTEM_PROCESSOR STREQUAL "ppc64le")
  SET(ORT_TARGET_PROCESSOR "ppc64")
ELSEIF(NOT ORT_TARGET_PROCESSOR MATCHES "^(x86(_64)?|arm64|riscv(32|64|128)|Hexagon|ppc64)$")
  SET(ORT_TARGET_PROCESSOR "${CMAKE_SYSTEM_PROCESSOR}")
ELSE()
  MESSAGE(FATAL_ERROR "Unrecognized CMAKE_SYSTEM_PROCESSOR value \"${CMAKE_SYSTEM_PROCESSOR}\"")
ENDIF()
MESSAGE(STATUS "Building for ORT_TARGET_PROCESSOR: ${ORT_TARGET_PROCESSOR}")

# KleidiAI is only used in Arm64 platform and not supported by MSVC, the details can be seen in
# https://github.com/google/XNNPACK/blob/3b3f7b8a6668f6ab3b6ce33b9f1d1fce971549d1/CMakeLists.txt#L206C82-L206C117
if(ORT_TARGET_PROCESSOR MATCHES "^arm64.*" AND NOT CMAKE_C_COMPILER_ID STREQUAL "MSVC")
  # kleidiAI use CMAKE_SYSTEM_PROCESSOR to determine whether includes aarch64/arm64 ukernels
  # https://gitlab.arm.com/kleidi/kleidiai/-/blob/main/CMakeLists.txt#L134
  set(CMAKE_SYSTEM_PROCESSOR arm64)
  FetchContent_Declare(kleidiai URL ${DEP_URL_kleidiai} URL_HASH SHA1=${DEP_SHA1_kleidiai})
  onnxruntime_fetchcontent_makeavailable(kleidiai)
  set(KLEIDIAI_SOURCE_DIR ${kleidiai_SOURCE_DIR})
endif()


FetchContent_Declare(googlexnnpack URL ${DEP_URL_googlexnnpack} URL_HASH SHA1=${DEP_SHA1_googlexnnpack}
                     PATCH_COMMAND ${Patch_EXECUTABLE} --binary --ignore-whitespace -p1 < ${PROJECT_SOURCE_DIR}/patches/xnnpack/AddEmscriptenAndIosSupport.patch
                    )
onnxruntime_fetchcontent_makeavailable(googlexnnpack)
set(XNNPACK_DIR ${googlexnnpack_SOURCE_DIR})
set(XNNPACK_INCLUDE_DIR ${XNNPACK_DIR}/include)

set(onnxruntime_EXTERNAL_LIBRARIES_XNNPACK XNNPACK microkernels-prod pthreadpool)
if(ORT_TARGET_PROCESSOR MATCHES "^arm64.*" AND NOT CMAKE_C_COMPILER_ID STREQUAL "MSVC")
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES_XNNPACK kleidiai)
endif()

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
