# This is a customization file to quickly inject necessary change into the build
# process pertaining to the constant sparse initializers.

# This is only supported on Linux although you can code on Windows
# With include headers avaiable.

option(onnxruntime_USE_SPARSE_LT "Build with CUDA cuSparse_lt support" OFF)

if(onnxruntime_USE_CUDA AND onnxruntime_USE_SPARSE_LT)
    list(APPEND ONNXRUNTIME_CUDA_LIBRARIES cusparseLt_static)
    if(NOT WIN32)
        link_directories(${onnxruntime_CUSPARSELT_HOME}/lib64)
    endif()
endif()
