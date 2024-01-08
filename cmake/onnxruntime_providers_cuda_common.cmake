# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

file(GLOB_RECURSE onnxruntime_providers_cuda_common_cc_srcs CONFIGURE_DEPENDS
        "${ONNXRUNTIME_ROOT}/core/providers/cuda/common/*.h"
        "${ONNXRUNTIME_ROOT}/core/providers/cuda/common/*.cc"
)
set(onnxruntime_providers_cuda_common_src ${onnxruntime_providers_cuda_common_cc_srcs})

# sets compile options and enables setting the same compile opts that the cuda lib uses with out linking all dependencies
function(config_cuda_compile_opts target)
    if (HAS_GUARD_CF)
        target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /guard:cf>")
    endif()
    if (HAS_QSPECTRE)
        target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /Qspectre>")
    endif()
    foreach(ORT_FLAG ${ORT_WARNING_FLAGS})
        target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler \"${ORT_FLAG}\">")
    endforeach()
    # CUDA 11.3+ supports parallel compilation
    # https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-guiding-compiler-driver-threads
    if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.3)
        option(onnxruntime_NVCC_THREADS "Number of threads that NVCC can use for compilation." 1)
        target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--threads \"${onnxruntime_NVCC_THREADS}\">")
    endif()
    if (UNIX)
        target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -Wno-reorder>"
                "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wno-reorder>")
        target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -Wno-error=sign-compare>"
                "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wno-error=sign-compare>")
    else()
        #mutex.cuh(91): warning C4834: discarding return value of function with 'nodiscard' attribute
        target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /wd4834>")
        target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /wd4127>")
    endif()
endfunction()

set(CUDAToolkit_ROOT ${onnxruntime_CUDA_HOME})
find_package(CUDAToolkit REQUIRED)
onnxruntime_add_static_library(onnxruntime_providers_cuda_common ${onnxruntime_providers_cuda_common_src})
target_link_libraries(onnxruntime_providers_cuda_common PRIVATE Boost::mp11 CUDA::cublasLt CUDA::cublas CUDA::cudart CUDA::cuda_driver cudnn ${ONNXRUNTIME_PROVIDERS_SHARED})
if(onnxruntime_CUDNN_HOME)
    target_include_directories(onnxruntime_providers_cuda_common PRIVATE ${onnxruntime_CUDNN_HOME}/include)
    target_link_directories(onnxruntime_providers_cuda_common PRIVATE ${onnxruntime_CUDNN_HOME}/lib ${onnxruntime_CUDNN_HOME}/lib/x64)
endif()
config_cuda_compile_opts(onnxruntime_providers_cuda_common)
onnxruntime_add_include_to_target(onnxruntime_providers_cuda_common onnxruntime_common onnxruntime_framework onnx onnx_proto ${PROTOBUF_LIB} flatbuffers::flatbuffers)

install(TARGETS onnxruntime_providers_cuda_common
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})
