# This is a customization file to quickly inject necessary change into the build
# process pertaining to the constant sparse initializers.

# This is only supported on Linux although you can code on Windows
# With include headers avaiable.
if(onnxruntime_USE_CUDA)
    target_link_libraries(onnxruntime_providers_cuda PUBLIC cusparse)
    if(onnxruntime_USE_SPARSE_LT)
        target_include_directories(onnxruntime_providers_cuda PRIVATE ${onnxruntime_CUSPARSELT_HOME}/include)
        target_compile_definitions(onnxruntime_providers_cuda PRIVATE -DUSE_CUSPARSELT)
    endif()
endif()