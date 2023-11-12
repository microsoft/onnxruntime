# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set(MLAS_SRC_DIR ${ONNXRUNTIME_ROOT}/core/mlas/lib)

#
# All hardware agnostic source files here
# hardware specific files would cause trouble in
# multi-target build
#
onnxruntime_add_static_library(onnxruntime_mlas
  ${MLAS_SRC_DIR}/platform.cpp
  ${MLAS_SRC_DIR}/threading.cpp
  ${MLAS_SRC_DIR}/sgemm.cpp
  ${MLAS_SRC_DIR}/halfgemm.cpp
  ${MLAS_SRC_DIR}/qgemm.cpp
  ${MLAS_SRC_DIR}/qdwconv.cpp
  ${MLAS_SRC_DIR}/convolve.cpp
  ${MLAS_SRC_DIR}/convsym.cpp
  ${MLAS_SRC_DIR}/pooling.cpp
  ${MLAS_SRC_DIR}/transpose.cpp
  ${MLAS_SRC_DIR}/reorder.cpp
  ${MLAS_SRC_DIR}/snchwc.cpp
  ${MLAS_SRC_DIR}/activate.cpp
  ${MLAS_SRC_DIR}/logistic.cpp
  ${MLAS_SRC_DIR}/tanh.cpp
  ${MLAS_SRC_DIR}/erf.cpp
  ${MLAS_SRC_DIR}/compute.cpp
  ${MLAS_SRC_DIR}/quantize.cpp
  ${MLAS_SRC_DIR}/qgemm_kernel_default.cpp
  ${MLAS_SRC_DIR}/qladd.cpp
  ${MLAS_SRC_DIR}/qlmul.cpp
  ${MLAS_SRC_DIR}/qpostprocessor.cpp
  ${MLAS_SRC_DIR}/qlgavgpool.cpp
  ${MLAS_SRC_DIR}/qdwconv_kernelsize.cpp
)

if (NOT onnxruntime_ORT_MINIMAL_BUILD)
  target_sources(onnxruntime_mlas PRIVATE
    ${MLAS_SRC_DIR}/q4_dq.cpp
    ${MLAS_SRC_DIR}/q4gemm.cpp
  )
endif()

set(ONNXRUNTIME_MLAS_LIBS onnxruntime_mlas)

#TODO: set MASM flags properly
function(setup_mlas_source_for_windows)

  #
  # Sources common for all platforms.
  #
  target_sources(onnxruntime_mlas PRIVATE
    ${MLAS_SRC_DIR}/activate_fp16.cpp
    ${MLAS_SRC_DIR}/dwconv.cpp
    ${MLAS_SRC_DIR}/pooling_fp16.cpp
  )

  #The onnxruntime_target_platform variable was added by Windows AI team in onnxruntime_common.cmake
  #Don't use it for other platforms.
  if((onnxruntime_target_platform STREQUAL "ARM64") OR (onnxruntime_target_platform STREQUAL "ARM64EC"))
    set(PREPROCESS_ARMASM_FLAGS "")
    set(ARMASM_FLAGS "")

    if(onnxruntime_target_platform STREQUAL "ARM64")
      target_sources(onnxruntime_mlas PRIVATE
        ${MLAS_SRC_DIR}/halfgemm_kernel_neon.cpp
        ${MLAS_SRC_DIR}/qgemm_kernel_neon.cpp
        ${MLAS_SRC_DIR}/qgemm_kernel_udot.cpp
        ${MLAS_SRC_DIR}/qgemm_kernel_sdot.cpp
      )

      set(mlas_platform_preprocess_srcs
        ${MLAS_SRC_DIR}/arm64/ConvSymS8KernelDot.asm
        ${MLAS_SRC_DIR}/arm64/ConvSymS8KernelDotLd64.asm
        ${MLAS_SRC_DIR}/arm64/ConvSymU8KernelDot.asm
        ${MLAS_SRC_DIR}/arm64/ConvSymS8KernelNeon.asm
        ${MLAS_SRC_DIR}/arm64/ConvSymU8KernelNeon.asm
        ${MLAS_SRC_DIR}/arm64/DepthwiseQConvSymS8KernelNeon.asm
        ${MLAS_SRC_DIR}/arm64/DepthwiseQConvSymU8KernelNeon.asm
        ${MLAS_SRC_DIR}/arm64/DepthwiseQConvKernelSize9Neon.asm
        ${MLAS_SRC_DIR}/arm64/HalfGemmKernelNeon.asm
        ${MLAS_SRC_DIR}/arm64/QgemmU8X8KernelNeon.asm
        ${MLAS_SRC_DIR}/arm64/QgemmS8S8KernelNeon.asm
        ${MLAS_SRC_DIR}/arm64/QgemmU8X8KernelUdot.asm
        ${MLAS_SRC_DIR}/arm64/QgemmS8S8KernelSdot.asm
        ${MLAS_SRC_DIR}/arm64/SgemmKernelNeon.asm
        ${MLAS_SRC_DIR}/arm64/SgemvKernelNeon.asm
        ${MLAS_SRC_DIR}/arm64/SymQgemmS8KernelNeon.asm
        ${MLAS_SRC_DIR}/arm64/SymQgemmS8KernelSDot.asm
        ${MLAS_SRC_DIR}/arm64/SymQgemmS8KernelSDotLd64.asm
      )
    else()
      target_sources(onnxruntime_mlas PRIVATE
        ${MLAS_SRC_DIR}/qgemm_kernel_neon.cpp
      )

      set(mlas_platform_preprocess_srcs
        ${MLAS_SRC_DIR}/arm64ec/QgemmU8X8KernelNeon.asm
        ${MLAS_SRC_DIR}/arm64ec/SgemmKernelNeon.asm
      )

      string(APPEND PREPROCESS_ARMASM_FLAGS " /arm64EC")
      string(APPEND ARMASM_FLAGS " -machine ARM64EC")
    endif()

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
      string(APPEND ARMASM_FLAGS " -g")
    endif()

    # Remove double quotes from flag strings.
    separate_arguments(PREPROCESS_ARMASM_FLAGS NATIVE_COMMAND "${PREPROCESS_ARMASM_FLAGS}")
    separate_arguments(ARMASM_FLAGS NATIVE_COMMAND "${ARMASM_FLAGS}")

    # Run the C precompiler on each input before the assembler.
    foreach(asm_filename ${mlas_platform_preprocess_srcs})
      get_filename_component(asm_filename_base ${asm_filename} NAME_WLE)
      set(preprocess_filename ${CMAKE_CURRENT_BINARY_DIR}/${asm_filename_base}.i)
      set(obj_filename ${CMAKE_CURRENT_BINARY_DIR}/${asm_filename_base}.obj)
      add_custom_command(
        OUTPUT ${obj_filename}
          COMMAND
              cl.exe ${PREPROCESS_ARMASM_FLAGS} /P ${asm_filename} /Fi${preprocess_filename}
          COMMAND
              armasm64.exe ${ARMASM_FLAGS} ${preprocess_filename} ${obj_filename}
        DEPENDS ${asm_filename}
        BYPRODUCTS ${preprocess_filename}
      )
      target_sources(onnxruntime_mlas PRIVATE ${obj_filename})
    endforeach()
  elseif(onnxruntime_target_platform STREQUAL "ARM")
    target_sources(onnxruntime_mlas PRIVATE
      ${MLAS_SRC_DIR}/arm/sgemmc.cpp
    )
    # it should be removed after Visual Stuio is upgraded to 17.7
    if (MSVC)
      add_compile_options("-d2SSAOptimizer-")
    endif()
  elseif(onnxruntime_target_platform STREQUAL "x64")

    file(GLOB_RECURSE mlas_platform_srcs_avx CONFIGURE_DEPENDS
      "${MLAS_SRC_DIR}/intrinsics/avx/*.cpp"
    )
    set_source_files_properties(${mlas_platform_srcs_avx} PROPERTIES COMPILE_FLAGS "/arch:AVX")

    file(GLOB_RECURSE mlas_platform_srcs_avx2 CONFIGURE_DEPENDS
      "${MLAS_SRC_DIR}/intrinsics/avx2/*.cpp"
    )
    set_source_files_properties(${mlas_platform_srcs_avx2} PROPERTIES COMPILE_FLAGS "/arch:AVX2")

    target_sources(onnxruntime_mlas PRIVATE
      ${MLAS_SRC_DIR}/dgemm.cpp
      ${mlas_platform_srcs_avx}
      ${mlas_platform_srcs_avx2}
      ${MLAS_SRC_DIR}/qgemm_kernel_amx.cpp
      ${MLAS_SRC_DIR}/qgemm_kernel_avx2.cpp
      ${MLAS_SRC_DIR}/qgemm_kernel_sse.cpp
      ${MLAS_SRC_DIR}/qgemm_kernel_sse41.cpp
      ${MLAS_SRC_DIR}/intrinsics/avx512/quantize_avx512f.cpp
      ${MLAS_SRC_DIR}/amd64/QgemmU8S8KernelAmx.asm
      ${MLAS_SRC_DIR}/amd64/QgemmU8S8KernelAvx2.asm
      ${MLAS_SRC_DIR}/amd64/QgemmU8U8KernelAvx2.asm
      ${MLAS_SRC_DIR}/amd64/QgemmU8X8KernelAvx2.asm
      ${MLAS_SRC_DIR}/amd64/QgemmU8X8KernelAvx512Core.asm
      ${MLAS_SRC_DIR}/amd64/QgemvU8S8KernelAvx2.asm
      ${MLAS_SRC_DIR}/amd64/QgemvU8S8KernelAvx512Core.asm
      ${MLAS_SRC_DIR}/amd64/QgemvU8S8KernelAvx512Vnni.asm
      ${MLAS_SRC_DIR}/amd64/QgemvU8S8KernelAvxVnni.asm
      ${MLAS_SRC_DIR}/amd64/ConvSymKernelAvx2.asm
      ${MLAS_SRC_DIR}/amd64/ConvSymKernelAvx512Core.asm
      ${MLAS_SRC_DIR}/amd64/DgemmKernelSse2.asm
      ${MLAS_SRC_DIR}/amd64/DgemmKernelAvx.asm
      ${MLAS_SRC_DIR}/amd64/DgemmKernelFma3.asm
      ${MLAS_SRC_DIR}/amd64/DgemmKernelAvx512F.asm
      ${MLAS_SRC_DIR}/amd64/SgemmKernelSse2.asm
      ${MLAS_SRC_DIR}/amd64/SgemmKernelAvx.asm
      ${MLAS_SRC_DIR}/amd64/SgemmKernelM1Avx.asm
      ${MLAS_SRC_DIR}/amd64/SgemmKernelFma3.asm
      ${MLAS_SRC_DIR}/amd64/SgemmKernelAvx512F.asm
      ${MLAS_SRC_DIR}/amd64/SconvKernelSse2.asm
      ${MLAS_SRC_DIR}/amd64/SconvKernelAvx.asm
      ${MLAS_SRC_DIR}/amd64/SconvKernelFma3.asm
      ${MLAS_SRC_DIR}/amd64/SconvKernelAvx512F.asm
      ${MLAS_SRC_DIR}/amd64/SpoolKernelSse2.asm
      ${MLAS_SRC_DIR}/amd64/SpoolKernelAvx.asm
      ${MLAS_SRC_DIR}/amd64/SpoolKernelAvx512F.asm
      ${MLAS_SRC_DIR}/amd64/sgemma.asm
      ${MLAS_SRC_DIR}/amd64/cvtfp16a.asm
      ${MLAS_SRC_DIR}/amd64/SoftmaxKernelAvx.asm
      ${MLAS_SRC_DIR}/amd64/TransKernelFma3.asm
      ${MLAS_SRC_DIR}/amd64/TransKernelAvx512F.asm
      ${MLAS_SRC_DIR}/amd64/LogisticKernelFma3.asm
      ${MLAS_SRC_DIR}/amd64/TanhKernelFma3.asm
      ${MLAS_SRC_DIR}/amd64/ErfKernelFma3.asm
    )
    if (NOT onnxruntime_ORT_MINIMAL_BUILD)
      target_sources(onnxruntime_mlas PRIVATE
        ${MLAS_SRC_DIR}/q4gemm_avx512.cpp
      )
    endif()

  else()
    target_sources(onnxruntime_mlas PRIVATE
      ${MLAS_SRC_DIR}/qgemm_kernel_sse.cpp
      ${MLAS_SRC_DIR}/qgemm_kernel_sse41.cpp
      ${MLAS_SRC_DIR}/i386/SgemmKernelSse2.asm
      ${MLAS_SRC_DIR}/i386/SgemmKernelAvx.asm
    )
  endif()
endfunction()

if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  if (onnxruntime_ENABLE_WEBASSEMBLY_SIMD)
    file(GLOB_RECURSE mlas_platform_srcs
      "${MLAS_SRC_DIR}/wasm_simd/*.cpp"
    )
    set(mlas_platform_srcs
      ${mlas_platform_srcs}
      ${MLAS_SRC_DIR}/qgemm_kernel_wasmsimd.cpp
    )
  else()
    file(GLOB_RECURSE mlas_platform_srcs
      "${MLAS_SRC_DIR}/scalar/*.cpp"
    )
  endif()
  target_sources(onnxruntime_mlas PRIVATE ${mlas_platform_srcs})
elseif(MSVC)
  setup_mlas_source_for_windows()
else()

    if(APPLE)
        get_target_property(ONNXRUNTIME_MLAS_OSX_ARCH onnxruntime_mlas OSX_ARCHITECTURES)

        if(NOT ONNXRUNTIME_MLAS_OSX_ARCH)
         set(ONNXRUNTIME_MLAS_OSX_ARCH ${CMAKE_HOST_SYSTEM_PROCESSOR})
        endif()
        foreach(OSX_ARCH ${ONNXRUNTIME_MLAS_OSX_ARCH})
        if (OSX_ARCH STREQUAL "arm64")
            set(ARM64 TRUE)
        elseif (OSX_ARCH STREQUAL "arm64e")
            set(ARM64 TRUE)
        elseif (OSX_ARCH STREQUAL "arm")
            set(ARM TRUE)
        elseif (OSX_ARCH STREQUAL "x86_64")
            set(X86_64 TRUE)
        elseif (OSX_ARCH STREQUAL "i386")
            set(X86 TRUE)
        endif()
        endforeach()
    elseif(ANDROID)
        if (CMAKE_ANDROID_ARCH_ABI STREQUAL "armeabi-v7a")
          set(ARM TRUE)
        elseif (CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a")
          set(ARM64 TRUE)
        elseif (CMAKE_ANDROID_ARCH_ABI STREQUAL "x86_64")
          set(X86_64 TRUE)
        elseif (CMAKE_ANDROID_ARCH_ABI STREQUAL "x86")
          set(X86 TRUE)
        endif()
    else()
        #Linux/FreeBSD/PowerPC/...
        #The value of CMAKE_SYSTEM_PROCESSOR should be from `uname -m`
        #Example values:
        #arm64v8/ubuntu -> aarch64
        #arm32v6/alpine -> armv7l
        #arm32v7/centos -> armv7l
        #ppc64le/debian -> ppc64le
        #s390x/ubuntu -> s390x
        #ppc64le/busybox -> ppc64le
        #arm64v8/ubuntu -> aarch64
        #Android: armv7-a aarch64 i686 x86_64
        #chasun: I don't think anyone uses 'arm64'
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "^arm64.*")
          set(ARM64 TRUE)
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^arm.*")
          set(ARM TRUE)
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64.*")
          set(ARM64 TRUE)
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(powerpc.*|ppc.*)")
          set(POWER TRUE)
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(i.86|x86?)$")
          set(X86 TRUE)
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64)$")
          set(X86_64 TRUE)
        endif()
    endif()

    if(APPLE)
      get_target_property(ONNXRUNTIME_MLAS_MACOSX_ARCH onnxruntime_mlas OSX_ARCHITECTURES)
    endif()
    list(LENGTH ONNXRUNTIME_MLAS_MACOSX_ARCH  ONNXRUNTIME_MLAS_MACOSX_ARCH_LENGH)
    if(ONNXRUNTIME_MLAS_MACOSX_ARCH_LENGH GREATER 1)
        set(ONNXRUNTIME_MLAS_MULTI_ARCH TRUE)
    endif()
    #If ONNXRUNTIME_MLAS_MULTI_ARCH is true, we need to go through every if branch below
    #and split MLAS to multiple static libraries.
    #Otherwise, it works like if(...) elseif(...) elseif(...) endif()
    set(MLAS_SOURCE_IS_NOT_SET 1)
    if(ARM)
        enable_language(ASM)

        set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} -mfpu=neon")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")

        set(mlas_platform_srcs
          ${MLAS_SRC_DIR}/aarch32/QgemmU8X8KernelNeon.S
          ${MLAS_SRC_DIR}/arm/sgemmc.cpp
          ${MLAS_SRC_DIR}/qgemm_kernel_neon.cpp
        )
        if(NOT ONNXRUNTIME_MLAS_MULTI_ARCH)
          set(MLAS_SOURCE_IS_NOT_SET 0)
        endif()
    endif()
    if(ARM64 AND MLAS_SOURCE_IS_NOT_SET )
        enable_language(ASM)
        set(mlas_platform_srcs
          ${MLAS_SRC_DIR}/aarch64/ConvSymS8KernelDot.S
          ${MLAS_SRC_DIR}/aarch64/ConvSymS8KernelDotLd64.S
          ${MLAS_SRC_DIR}/aarch64/ConvSymU8KernelDot.S
          ${MLAS_SRC_DIR}/aarch64/ConvSymS8KernelNeon.S
          ${MLAS_SRC_DIR}/aarch64/ConvSymU8KernelNeon.S
          ${MLAS_SRC_DIR}/aarch64/DepthwiseQConvSymS8KernelNeon.S
          ${MLAS_SRC_DIR}/aarch64/DepthwiseQConvSymU8KernelNeon.S
          ${MLAS_SRC_DIR}/aarch64/DepthwiseQConvKernelSize9Neon.S
          ${MLAS_SRC_DIR}/aarch64/QgemmU8X8KernelNeon.S
          ${MLAS_SRC_DIR}/aarch64/QgemmS8S8KernelNeon.S
          ${MLAS_SRC_DIR}/aarch64/QgemmU8X8KernelUdot.S
          ${MLAS_SRC_DIR}/aarch64/QgemmS8S8KernelSdot.S
          ${MLAS_SRC_DIR}/aarch64/SgemmKernelNeon.S
          ${MLAS_SRC_DIR}/aarch64/SgemvKernelNeon.S
          ${MLAS_SRC_DIR}/aarch64/SymQgemmS8KernelNeon.S
          ${MLAS_SRC_DIR}/aarch64/SymQgemmS8KernelSdot.S
          ${MLAS_SRC_DIR}/aarch64/SymQgemmS8KernelSdotLd64.S
          ${MLAS_SRC_DIR}/qgemm_kernel_neon.cpp
          ${MLAS_SRC_DIR}/qgemm_kernel_udot.cpp
          ${MLAS_SRC_DIR}/qgemm_kernel_sdot.cpp
        )
        if (NOT APPLE)
          set(mlas_platform_srcs
            ${mlas_platform_srcs}
            ${MLAS_SRC_DIR}/aarch64/HalfGemmKernelNeon.S
            ${MLAS_SRC_DIR}/aarch64/QgemmS8S8KernelSmmla.S
            ${MLAS_SRC_DIR}/aarch64/QgemmU8X8KernelUmmla.S
            ${MLAS_SRC_DIR}/activate_fp16.cpp
            ${MLAS_SRC_DIR}/dwconv.cpp
            ${MLAS_SRC_DIR}/halfgemm_kernel_neon.cpp
            ${MLAS_SRC_DIR}/pooling_fp16.cpp
            ${MLAS_SRC_DIR}/qgemm_kernel_smmla.cpp
            ${MLAS_SRC_DIR}/qgemm_kernel_ummla.cpp
          )
          set_source_files_properties(${MLAS_SRC_DIR}/aarch64/HalfGemmKernelNeon.S PROPERTIES COMPILE_FLAGS " -march=armv8.2-a+fp16 ")
          set_source_files_properties(${MLAS_SRC_DIR}/aarch64/QgemmS8S8KernelSmmla.S PROPERTIES COMPILE_FLAGS " -march=armv8.2-a+i8mm ")
          set_source_files_properties(${MLAS_SRC_DIR}/aarch64/QgemmU8X8KernelUmmla.S PROPERTIES COMPILE_FLAGS " -march=armv8.2-a+i8mm ")
          set_source_files_properties(${MLAS_SRC_DIR}/activate_fp16.cpp PROPERTIES COMPILE_FLAGS " -march=armv8.2-a+fp16 ")
          set_source_files_properties(${MLAS_SRC_DIR}/dwconv.cpp PROPERTIES COMPILE_FLAGS " -march=armv8.2-a+fp16 ")
          set_source_files_properties(${MLAS_SRC_DIR}/pooling_fp16.cpp PROPERTIES COMPILE_FLAGS " -march=armv8.2-a+fp16 ")
        endif()

        if(ONNXRUNTIME_MLAS_MULTI_ARCH)
            onnxruntime_add_static_library(onnxruntime_mlas_arm64 ${mlas_platform_srcs})
            set_target_properties(onnxruntime_mlas_arm64 PROPERTIES OSX_ARCHITECTURES "arm64")
            list(APPEND ONNXRUNTIME_MLAS_LIBS onnxruntime_mlas_arm64)
            set(mlas_platform_srcs )
        else()
            set(MLAS_SOURCE_IS_NOT_SET 0)
        endif()
    endif()
    if(POWER AND MLAS_SOURCE_IS_NOT_SET)
        set(mlas_platform_srcs
          ${MLAS_SRC_DIR}/power/SgemmKernelPower.cpp
          ${MLAS_SRC_DIR}/dgemm.cpp
          ${MLAS_SRC_DIR}/power/DgemmKernelPower.cpp
          ${MLAS_SRC_DIR}/power/QuantizePower.cpp
        )
        set_source_files_properties(${MLAS_SRC_DIR}/power/SgemmKernelPower.cpp PROPERTIES COMPILE_FLAGS "-DSINGLE")

        check_cxx_compiler_flag("-mcpu=power9" HAS_POWER9)
        if (HAS_POWER9)
          set(mlas_platform_srcs
            ${mlas_platform_srcs}
            ${MLAS_SRC_DIR}/power/QuantizePowerVSX.cpp
          )
          set_source_files_properties(${MLAS_SRC_DIR}/power/QuantizePowerVSX.cpp PROPERTIES COMPILE_FLAGS "-mcpu=power9")
        endif()

        check_cxx_compiler_flag("-mcpu=power10" HAS_POWER10)
        if(HAS_POWER10)
          set(CMAKE_REQUIRED_FLAGS "-mcpu=power10")
          check_cxx_source_compiles("
            #include <altivec.h>
            int main() {
              __vector_quad acc0;
              __builtin_mma_xxsetaccz (&acc0);
              return 0;
            }"
            COMPILES_P10
          )
          if(COMPILES_P10)
            check_cxx_source_compiles("
              #include <sys/auxv.h>
              int main() {
                unsigned long hwcap2 = getauxval(AT_HWCAP2);
                bool HasP10 = ((hwcap2 & PPC_FEATURE2_MMA) && (hwcap2 & PPC_FEATURE2_ARCH_3_1));
                return 0;
              }"
              HAS_P10_RUNTIME
            )
            if (HAS_P10_RUNTIME)
              set_source_files_properties(${MLAS_SRC_DIR}/platform.cpp PROPERTIES COMPILE_FLAGS "-DPOWER10")
              set_source_files_properties(${MLAS_SRC_DIR}/qgemm.cpp PROPERTIES COMPILE_FLAGS "-DPOWER10")
            endif()
            set(mlas_platform_srcs_power10
              ${MLAS_SRC_DIR}/power/SgemmKernelPOWER10.cpp
              ${MLAS_SRC_DIR}/power/DgemmKernelPOWER10.cpp
              ${MLAS_SRC_DIR}/power/qgemm_kernel_power10.cpp
            )
            set_source_files_properties(${MLAS_SRC_DIR}/power/SgemmKernelPOWER10.cpp PROPERTIES COMPILE_FLAGS "-O2 -mcpu=power10 -DSINGLE")
            set_source_files_properties(${MLAS_SRC_DIR}/power/DgemmKernelPOWER10.cpp PROPERTIES COMPILE_FLAGS "-O2 -mcpu=power10")
            set_source_files_properties(${MLAS_SRC_DIR}/power/qgemm_kernel_power10.cpp PROPERTIES COMPILE_FLAGS "-O3 -mcpu=power10")
            set(mlas_platform_srcs
              ${mlas_platform_srcs}
              ${mlas_platform_srcs_power10}
            )
          endif()
        endif()
        if(NOT ONNXRUNTIME_MLAS_MULTI_ARCH)
          set(MLAS_SOURCE_IS_NOT_SET 0)
        endif()
    endif()
    if(X86 AND MLAS_SOURCE_IS_NOT_SET)
        enable_language(ASM)

        set(mlas_platform_srcs_sse2
          ${MLAS_SRC_DIR}/qgemm_kernel_sse.cpp
          ${MLAS_SRC_DIR}/x86/SgemmKernelSse2.S
        )
        set_source_files_properties(${mlas_platform_srcs_sse2} PROPERTIES COMPILE_FLAGS "-msse2")

        set(mlas_platform_srcs_avx
          ${MLAS_SRC_DIR}/x86/SgemmKernelAvx.S
        )
        set_source_files_properties(${mlas_platform_srcs_avx} PROPERTIES COMPILE_FLAGS "-mavx")

        set(mlas_platform_srcs
          ${mlas_platform_srcs_sse2}
          ${mlas_platform_srcs_avx}
        )

        # In r23, NDK remove __x86.get_pc_thunk.* from libatomic. Add our own
        # implementation to avoid external dependency.
        if(ANDROID)
          set(mlas_platform_srcs
            ${mlas_platform_srcs}
            ${MLAS_SRC_DIR}/x86/x86.get_pc_thunk.S
          )
        endif()

        if(NOT ONNXRUNTIME_MLAS_MULTI_ARCH)
          set(MLAS_SOURCE_IS_NOT_SET 0)
        endif()
    endif()
    if(X86_64 AND MLAS_SOURCE_IS_NOT_SET)
        enable_language(ASM)

        # Forward the flags for the minimum target platform version from the C
        # compiler to the assembler. This works around CMakeASMCompiler.cmake.in
        # not including the logic to set this flag for the assembler.
        set(CMAKE_ASM${ASM_DIALECT}_OSX_DEPLOYMENT_TARGET_FLAG "${CMAKE_C_OSX_DEPLOYMENT_TARGET_FLAG}")

        # The LLVM assembler does not support the .arch directive to enable instruction
        # set extensions and also doesn't support AVX-512F instructions without
        # turning on support via command-line option. Group the sources by the
        # instruction set extension and explicitly set the compiler flag as appropriate.

        set(mlas_platform_srcs_sse2
          ${MLAS_SRC_DIR}/qgemm_kernel_sse.cpp
          ${MLAS_SRC_DIR}/x86_64/DgemmKernelSse2.S
          ${MLAS_SRC_DIR}/x86_64/SgemmKernelSse2.S
          ${MLAS_SRC_DIR}/x86_64/SgemmTransposePackB16x4Sse2.S
          ${MLAS_SRC_DIR}/x86_64/SconvKernelSse2.S
          ${MLAS_SRC_DIR}/x86_64/SpoolKernelSse2.S
        )
        set_source_files_properties(${mlas_platform_srcs_sse2} PROPERTIES COMPILE_FLAGS "-msse2")

        set(mlas_platform_srcs_avx
          ${MLAS_SRC_DIR}/x86_64/DgemmKernelAvx.S
          ${MLAS_SRC_DIR}/x86_64/SgemmKernelAvx.S
          ${MLAS_SRC_DIR}/x86_64/SgemmKernelM1Avx.S
          ${MLAS_SRC_DIR}/x86_64/SgemmKernelM1TransposeBAvx.S
          ${MLAS_SRC_DIR}/x86_64/SgemmTransposePackB16x4Avx.S
          ${MLAS_SRC_DIR}/x86_64/SconvKernelAvx.S
          ${MLAS_SRC_DIR}/x86_64/SpoolKernelAvx.S
          ${MLAS_SRC_DIR}/x86_64/SoftmaxKernelAvx.S
          ${MLAS_SRC_DIR}/intrinsics/avx/min_max_elements.cpp
        )
        set_source_files_properties(${mlas_platform_srcs_avx} PROPERTIES COMPILE_FLAGS "-mavx")

        set(mlas_platform_srcs_avx2
          ${MLAS_SRC_DIR}/x86_64/QgemmU8S8KernelAvx2.S
          ${MLAS_SRC_DIR}/x86_64/QgemvU8S8KernelAvx2.S
          ${MLAS_SRC_DIR}/x86_64/QgemmU8U8KernelAvx2.S
          ${MLAS_SRC_DIR}/x86_64/QgemvU8S8KernelAvxVnni.S
          ${MLAS_SRC_DIR}/x86_64/QgemmU8X8KernelAvx2.S
          ${MLAS_SRC_DIR}/x86_64/ConvSymKernelAvx2.S
          ${MLAS_SRC_DIR}/x86_64/DgemmKernelFma3.S
          ${MLAS_SRC_DIR}/x86_64/SgemmKernelFma3.S
          ${MLAS_SRC_DIR}/x86_64/SconvKernelFma3.S
          ${MLAS_SRC_DIR}/x86_64/TransKernelFma3.S
          ${MLAS_SRC_DIR}/x86_64/LogisticKernelFma3.S
          ${MLAS_SRC_DIR}/x86_64/TanhKernelFma3.S
          ${MLAS_SRC_DIR}/x86_64/ErfKernelFma3.S
          ${MLAS_SRC_DIR}/intrinsics/avx2/qladd_avx2.cpp
          ${MLAS_SRC_DIR}/intrinsics/avx2/qdwconv_avx2.cpp
        )
        set_source_files_properties(${mlas_platform_srcs_avx2} PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")

        set(mlas_platform_srcs_avx512f
          ${MLAS_SRC_DIR}/x86_64/DgemmKernelAvx512F.S
          ${MLAS_SRC_DIR}/x86_64/SgemmKernelAvx512F.S
          ${MLAS_SRC_DIR}/x86_64/SconvKernelAvx512F.S
          ${MLAS_SRC_DIR}/x86_64/SpoolKernelAvx512F.S
          ${MLAS_SRC_DIR}/x86_64/TransKernelAvx512F.S
          ${MLAS_SRC_DIR}/intrinsics/avx512/quantize_avx512f.cpp
        )
        set_source_files_properties(${mlas_platform_srcs_avx512f} PROPERTIES COMPILE_FLAGS "-mavx512f")

        set(mlas_platform_srcs_avx512core
          ${MLAS_SRC_DIR}/x86_64/QgemvU8S8KernelAvx512Core.S
          ${MLAS_SRC_DIR}/x86_64/QgemvU8S8KernelAvx512Vnni.S
          ${MLAS_SRC_DIR}/x86_64/QgemmU8X8KernelAvx512Core.S
          ${MLAS_SRC_DIR}/x86_64/ConvSymKernelAvx512Core.S
        )
        set_source_files_properties(${mlas_platform_srcs_avx512core} PROPERTIES COMPILE_FLAGS "-mavx512bw -mavx512dq -mavx512vl")

        set(mlas_platform_srcs
          ${MLAS_SRC_DIR}/activate_fp16.cpp
          ${MLAS_SRC_DIR}/dwconv.cpp
          ${MLAS_SRC_DIR}/dgemm.cpp
          ${MLAS_SRC_DIR}/pooling_fp16.cpp
          ${MLAS_SRC_DIR}/qgemm_kernel_avx2.cpp
          ${mlas_platform_srcs_sse2}
          ${mlas_platform_srcs_avx}
          ${mlas_platform_srcs_avx2}
          ${mlas_platform_srcs_avx512f}
          ${mlas_platform_srcs_avx512core}
        )

        if (NOT onnxruntime_ORT_MINIMAL_BUILD)
          set(mlas_platform_srcs
            ${mlas_platform_srcs}
            ${MLAS_SRC_DIR}/q4gemm_avx512.cpp
          )
          set_source_files_properties(${MLAS_SRC_DIR}/q4gemm_avx512.cpp PROPERTIES COMPILE_FLAGS "-mfma -mavx512vnni -mavx512bw -mavx512dq -mavx512vl -mavx512f")
        endif()
        if(NOT APPLE)
          set(mlas_platform_srcs
            ${mlas_platform_srcs}
	        ${MLAS_SRC_DIR}/x86_64/QgemmU8S8KernelAmxCommon.S
            ${MLAS_SRC_DIR}/qgemm_kernel_amx.cpp
            ${MLAS_SRC_DIR}/x86_64/QgemmU8S8KernelAmx.S
            )
          set_source_files_properties(${MLAS_SRC_DIR}/qgemm_kernel_amx.cpp PROPERTIES COMPILE_FLAGS "-mavx2 -mavx512bw -mavx512dq -mavx512vl -mavx512f")
          set_source_files_properties(${MLAS_SRC_DIR}/x86_64/QgemmU8S8KernelAmx.S PROPERTIES COMPILE_FLAGS "-mavx2 -mavx512bw -mavx512dq -mavx512vl -mavx512f")
	    endif()

        if(ONNXRUNTIME_MLAS_MULTI_ARCH)
          onnxruntime_add_static_library(onnxruntime_mlas_x86_64 ${mlas_platform_srcs})
          set_target_properties(onnxruntime_mlas_x86_64 PROPERTIES OSX_ARCHITECTURES "x86_64")
          list(APPEND ONNXRUNTIME_MLAS_LIBS onnxruntime_mlas_x86_64)
          set(mlas_platform_srcs )
        else()
          set(MLAS_SOURCE_IS_NOT_SET 0)
        endif()
    endif()
    if(NOT ONNXRUNTIME_MLAS_MULTI_ARCH AND MLAS_SOURCE_IS_NOT_SET)
        file(GLOB_RECURSE mlas_platform_srcs
          "${MLAS_SRC_DIR}/scalar/*.cpp")
    endif()
    target_sources(onnxruntime_mlas PRIVATE ${mlas_platform_srcs})
endif()

foreach(mlas_target ${ONNXRUNTIME_MLAS_LIBS})
    target_include_directories(${mlas_target} PRIVATE ${ONNXRUNTIME_ROOT}/core/mlas/inc ${MLAS_SRC_DIR})
    onnxruntime_add_include_to_target(${mlas_target} ${GSL_TARGET})
endforeach()
set_target_properties(onnxruntime_mlas PROPERTIES FOLDER "ONNXRuntime")
if (WIN32)
  target_compile_options(onnxruntime_mlas PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/wd6385>" "$<$<COMPILE_LANGUAGE:CXX>:/wd4127>")
  if (onnxruntime_ENABLE_STATIC_ANALYSIS)
    target_compile_options(onnxruntime_mlas PRIVATE  "$<$<COMPILE_LANGUAGE:CXX>:/analyze:stacksize 131072>")
  endif()
endif()

if (NOT onnxruntime_BUILD_SHARED_LIB)
    install(TARGETS onnxruntime_mlas
            ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
            FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()


if (NOT onnxruntime_ORT_MINIMAL_BUILD)

  #
  # Command line tool for quantization and de-quantization of 2-D fp32 tensors
  # based on block-wise quantization of int4
  #

  onnxruntime_add_executable(onnxruntime_mlas_q4dq
    ${MLAS_SRC_DIR}/q4_dq_cli.cpp
  )
  target_include_directories(onnxruntime_mlas_q4dq PRIVATE ${ONNXRUNTIME_ROOT}/core/mlas/inc ${MLAS_SRC_DIR})
  set_target_properties(onnxruntime_mlas_q4dq PROPERTIES FOLDER "ONNXRuntimeTest")

  target_link_libraries(onnxruntime_mlas_q4dq PRIVATE ${ONNXRUNTIME_MLAS_LIBS} onnxruntime_common)
  if (CPUINFO_SUPPORTED AND NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    target_link_libraries(onnxruntime_mlas_q4dq PRIVATE cpuinfo)
  endif()
  if(NOT WIN32)
    target_link_libraries(onnxruntime_mlas_q4dq PRIVATE nsync::nsync_cpp ${CMAKE_DL_LIBS})
  endif()
  if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    target_link_libraries(onnxruntime_mlas_q4dq PRIVATE ${android_shared_libs})
  endif()

  if(WIN32)
    target_link_libraries(onnxruntime_mlas_q4dq PRIVATE debug Dbghelp Advapi32)
  endif()
  if (onnxruntime_LINK_LIBATOMIC)
    target_link_libraries(onnxruntime_mlas_q4dq PRIVATE atomic)
  endif()
  target_link_libraries(onnxruntime_mlas_q4dq PRIVATE Threads::Threads)

  if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    if (onnxruntime_ENABLE_WEBASSEMBLY_THREADS)
      set_target_properties(onnxruntime_mlas_q4dq PROPERTIES LINK_FLAGS "-s ALLOW_MEMORY_GROWTH=1 -s PROXY_TO_PTHREAD=1 -s EXIT_RUNTIME=1")
    else()
      set_target_properties(onnxruntime_mlas_q4dq PROPERTIES LINK_FLAGS "-s ALLOW_MEMORY_GROWTH=1")
    endif()
  endif()

endif()
