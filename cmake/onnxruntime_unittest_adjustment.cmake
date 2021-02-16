# cuSparse experiments customization
if(onnxruntime_USE_CUDA)
    # target_link_libraries(onnxruntime_test_all PRIVATE cusparse)
    if(onnxruntime_USE_SPARSE_LT)
      target_include_directories(onnxruntime_test_all PRIVATE ${onnxruntime_CUSPARSELT_HOME}/include)
      target_compile_definitions(onnxruntime_test_all PRIVATE -DUSE_CUSPARSELT)
    endif()
endif()
