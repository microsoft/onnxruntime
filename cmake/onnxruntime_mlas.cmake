# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set(mlas_common_srcs
  ${ONNXRUNTIME_ROOT}/core/mlas/lib/platform.cpp
  ${ONNXRUNTIME_ROOT}/core/mlas/lib/threading.cpp
  ${ONNXRUNTIME_ROOT}/core/mlas/lib/qgemm.cpp
  ${ONNXRUNTIME_ROOT}/core/mlas/lib/sgemm.cpp
  ${ONNXRUNTIME_ROOT}/core/mlas/lib/convolve.cpp
  ${ONNXRUNTIME_ROOT}/core/mlas/lib/pooling.cpp
  ${ONNXRUNTIME_ROOT}/core/mlas/lib/reorder.cpp
  ${ONNXRUNTIME_ROOT}/core/mlas/lib/snchwc.cpp
  ${ONNXRUNTIME_ROOT}/core/mlas/lib/activate.cpp
  ${ONNXRUNTIME_ROOT}/core/mlas/lib/logistic.cpp
  ${ONNXRUNTIME_ROOT}/core/mlas/lib/tanh.cpp
  ${ONNXRUNTIME_ROOT}/core/mlas/lib/erf.cpp
)

if(MSVC)
  if(CMAKE_GENERATOR_PLATFORM STREQUAL "ARM64")
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
  elseif(CMAKE_GENERATOR_PLATFORM STREQUAL "ARM" OR CMAKE_GENERATOR MATCHES "ARM")
    set(mlas_platform_srcs
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/arm/sgemmc.cpp
    )
  elseif(CMAKE_GENERATOR_PLATFORM STREQUAL "x64" OR CMAKE_GENERATOR MATCHES "Win64")
    enable_language(ASM_MASM)

    set(mlas_platform_srcs
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/QgemmU8U8KernelAvx2.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/QgemmU8U8KernelAvx512BW.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/QgemmU8U8KernelAvx512Vnni.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/SgemmKernelSse2.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/SgemmKernelAvx.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/SgemmKernelFma3.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/SgemmKernelAvx512F.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/SconvKernelSse2.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/SconvKernelAvx.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/SconvKernelFma3.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/SconvKernelAvx512F.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/SpoolKernelSse2.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/SpoolKernelAvx.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/SpoolKernelAvx512F.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/sgemma.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/cvtfp16a.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/LogisticKernelFma3.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/TanhKernelFma3.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/ErfKernelFma3.asm
    )
  else()
    enable_language(ASM_MASM)

    set(CMAKE_ASM_MASM_FLAGS "${CMAKE_ASM_MASM_FLAGS} /safeseh")

    set(mlas_platform_srcs
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/i386/sgemma.asm
    )
  endif()
else()
  if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    if (CMAKE_ANDROID_ARCH_ABI STREQUAL "armeabi-v7a")
      set(ARM TRUE)
    elseif (CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a")
      set(ARM TRUE) # Android NDK fails to compile sgemma.s
    elseif (CMAKE_ANDROID_ARCH_ABI STREQUAL "x86_64")
      set(X86_64 TRUE)
    elseif (CMAKE_ANDROID_ARCH_ABI STREQUAL "x86")
      set(X86 TRUE)
    endif()
  else()
    execute_process(
      COMMAND ${CMAKE_C_COMPILER} -dumpmachine
      OUTPUT_VARIABLE dumpmachine_output
      ERROR_QUIET
    )
    if(dumpmachine_output MATCHES "^arm.*")
      set(ARM TRUE)
    elseif(dumpmachine_output MATCHES "^aarch64.*")
      set(ARM64 TRUE)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(i.86|x86?)$")
      set(X86 TRUE)
    elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
      set(X86_64 TRUE)
    endif()
  endif()

  if(ARM)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")

    set(mlas_platform_srcs
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/arm/sgemmc.cpp
    )
  elseif(ARM64)
    enable_language(ASM)

    set(mlas_platform_srcs
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/aarch64/sgemma.s
    )
  elseif(X86)
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
  elseif(X86_64)
    enable_language(ASM)

    # The LLVM assembler does not support the .arch directive to enable instruction
    # set extensions and also doesn't support AVX-512F instructions without
    # turning on support via command-line option. Group the sources by the
    # instruction set extension and explicitly set the compiler flag as appropriate.

    set(mlas_platform_srcs_sse2
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/SgemmKernelSse2.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/SgemmTransposePackB16x4Sse2.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/SconvKernelSse2.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/SpoolKernelSse2.S
    )
    set_source_files_properties(${mlas_platform_srcs_sse2} PROPERTIES COMPILE_FLAGS "-msse2")

    set(mlas_platform_srcs_avx
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/SgemmKernelAvx.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/SgemmKernelM1Avx.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/SgemmKernelM1TransposeBAvx.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/SgemmTransposePackB16x4Avx.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/SconvKernelAvx.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/SpoolKernelAvx.S
    )
    set_source_files_properties(${mlas_platform_srcs_avx} PROPERTIES COMPILE_FLAGS "-mavx")

    set(mlas_platform_srcs_avx2
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/QgemmU8U8KernelAvx2.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/SgemmKernelFma3.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/SconvKernelFma3.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/LogisticKernelFma3.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/TanhKernelFma3.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/ErfKernelFma3.S
    )
    set_source_files_properties(${mlas_platform_srcs_avx2} PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")

    set(mlas_platform_srcs_avx512f
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/SgemmKernelAvx512F.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/SconvKernelAvx512F.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/SpoolKernelAvx512F.S
    )
    set_source_files_properties(${mlas_platform_srcs_avx512f} PROPERTIES COMPILE_FLAGS "-mavx512f")

    set(mlas_platform_srcs_avx512bw
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/QgemmU8U8KernelAvx512BW.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/QgemmU8U8KernelAvx512Vnni.S
    )
    set_source_files_properties(${mlas_platform_srcs_avx512bw} PROPERTIES COMPILE_FLAGS "-mavx512bw")

    set(mlas_platform_srcs
      ${mlas_platform_srcs_sse2}
      ${mlas_platform_srcs_avx}
      ${mlas_platform_srcs_avx2}
      ${mlas_platform_srcs_avx512f}
      ${mlas_platform_srcs_avx512bw}
    )
  endif()
endif()

add_library(onnxruntime_mlas STATIC ${mlas_common_srcs} ${mlas_platform_srcs})
target_include_directories(onnxruntime_mlas PRIVATE ${ONNXRUNTIME_ROOT}/core/mlas/inc ${ONNXRUNTIME_ROOT}/core/mlas/lib)
set_target_properties(onnxruntime_mlas PROPERTIES FOLDER "ONNXRuntime")
