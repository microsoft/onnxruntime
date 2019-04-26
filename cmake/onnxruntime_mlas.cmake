# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set(mlas_common_srcs
  ${ONNXRUNTIME_ROOT}/core/mlas/lib/platform.cpp
  ${ONNXRUNTIME_ROOT}/core/mlas/lib/threading.cpp
  ${ONNXRUNTIME_ROOT}/core/mlas/lib/sgemm.cpp
  ${ONNXRUNTIME_ROOT}/core/mlas/lib/convolve.cpp
  ${ONNXRUNTIME_ROOT}/core/mlas/lib/pooling.cpp
  ${ONNXRUNTIME_ROOT}/core/mlas/lib/activate.cpp
  ${ONNXRUNTIME_ROOT}/core/mlas/lib/logistic.cpp
  ${ONNXRUNTIME_ROOT}/core/mlas/lib/tanh.cpp
)

if(MSVC)

  if(CMAKE_GENERATOR_PLATFORM STREQUAL "ARM")

    set(mlas_platform_srcs
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/arm/sgemmc.cpp
    )

  elseif(CMAKE_GENERATOR_PLATFORM STREQUAL "ARM64")

    set(asm_filename ${ONNXRUNTIME_ROOT}/core/mlas/lib/arm64/sgemma.asm)
    set(pre_filename ${CMAKE_CURRENT_BINARY_DIR}/sgemma.i)
    set(obj_filename ${CMAKE_CURRENT_BINARY_DIR}/sgemma.obj)

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
      set(ARMASM_FLAGS "-g")
    else()
      set(ARMASM_FLAGS "")
    endif()

    add_custom_command(
      OUTPUT ${obj_filename}
        COMMAND
            cl.exe /P ${asm_filename}
        COMMAND
            armasm64.exe ${ARMASM_FLAGS} ${pre_filename} ${obj_filename}
    )

    set(mlas_platform_srcs ${obj_filename})

  elseif(CMAKE_GENERATOR_PLATFORM STREQUAL "Win32")

    enable_language(ASM_MASM)

    set(CMAKE_ASM_MASM_FLAGS "${CMAKE_ASM_MASM_FLAGS} /safeseh")

    set(mlas_platform_srcs
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/i386/sgemma.asm
    )

  elseif(CMAKE_GENERATOR_PLATFORM STREQUAL "x64")

    enable_language(ASM_MASM)

    set(mlas_platform_srcs
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/SgemmKernelSse2.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/SgemmKernelAvx.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/SgemmKernelFma3.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/SgemmKernelAvx512F.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/sgemma.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/cvtfp16a.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/LogisticKernelFma3.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/TanhKernelFma3.asm
    )

  endif()

elseif(CMAKE_SYSTEM_NAME STREQUAL "Android")

  if(CMAKE_ANDROID_ARCH_ABI MATCHES "^arm.*")

    if(CMAKE_ANDROID_ARCH_ABI STREQUAL "armeabi-v7a")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")
    endif()

    set(mlas_platform_srcs
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/arm/sgemmc.cpp
    )

  else()

    message(FATAL_ERROR "Android build is not supported on non-ARM platform now")

  endif()

else()

  execute_process(
    COMMAND ${CMAKE_C_COMPILER} -dumpmachine
    OUTPUT_VARIABLE dumpmachine_output
    ERROR_QUIET
  )

  if(dumpmachine_output MATCHES "^arm.*")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")

    set(mlas_platform_srcs
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/arm/sgemmc.cpp
    )

  elseif(dumpmachine_output MATCHES "^aarch64.*")

    enable_language(ASM)

    set(mlas_platform_srcs
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/aarch64/sgemma.s
    )

  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(i.86|x86?)$")

    enable_language(ASM)

    set(mlas_platform_srcs_sse2
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86/SgemmKernelSse2.S
    )
    set_source_files_properties(${mlas_platform_srcs_sse2} PROPERTIES COMPILE_FLAGS "-msse2")

    set(mlas_platform_srcs_avx
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86/SgemmKernelAvx.S
    )
    set_source_files_properties(${mlas_platform_srcs_avx} PROPERTIES COMPILE_FLAGS "-mavx")

    set(mlas_platform_srcs
      ${mlas_platform_srcs_sse2}
      ${mlas_platform_srcs_avx}
    )

  elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")

    enable_language(ASM)

    # The LLVM assmebler does not support the .arch directive to enable instruction
    # set extensions and also doesn't support AVX-512F instructions without
    # turning on support via command-line option. Group the sources by the
    # instruction set extension and explicitly set the compiler flag as appropriate.

    set(mlas_platform_srcs_sse2
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/SgemmKernelSse2.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/SgemmTransposePackB16x4Sse2.S
    )
    set_source_files_properties(${mlas_platform_srcs_sse2} PROPERTIES COMPILE_FLAGS "-msse2")

    set(mlas_platform_srcs_avx
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/SgemmKernelAvx.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/SgemmKernelM1Avx.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/SgemmKernelM1TransposeBAvx.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/SgemmTransposePackB16x4Avx.S
    )
    set_source_files_properties(${mlas_platform_srcs_avx} PROPERTIES COMPILE_FLAGS "-mavx")

    set(mlas_platform_srcs_avx2
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/SgemmKernelFma3.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/LogisticKernelFma3.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/TanhKernelFma3.S
    )
    set_source_files_properties(${mlas_platform_srcs_avx2} PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")

    set(mlas_platform_srcs_avx512f
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/SgemmKernelAvx512F.S
    )
    set_source_files_properties(${mlas_platform_srcs_avx512f} PROPERTIES COMPILE_FLAGS "-mavx512f")

    set(mlas_platform_srcs
      ${mlas_platform_srcs_sse2}
      ${mlas_platform_srcs_avx}
      ${mlas_platform_srcs_avx2}
      ${mlas_platform_srcs_avx512f}
    )

  endif()

endif()

add_library(onnxruntime_mlas STATIC ${mlas_common_srcs} ${mlas_platform_srcs})
target_include_directories(onnxruntime_mlas PRIVATE ${ONNXRUNTIME_ROOT}/core/mlas/inc ${ONNXRUNTIME_ROOT}/core/mlas/lib)
set_target_properties(onnxruntime_mlas PROPERTIES FOLDER "ONNXRuntime")
