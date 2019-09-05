if(onnxruntime_BUILD_JNI)
  add_library(onnxruntime-jni
      SHARED
      ${REPO_ROOT}/java/java_api.cpp
      ${REPO_ROOT}/java/jni_helper.h)
  if (NOT CMAKE_SYSTEM_NAME STREQUAL "Android")
    find_package(JNI REQUIRED)
    target_include_directories(onnxruntime-jni
        PUBLIC
        ${JNI_INCLUDE_DIRS})

    target_link_libraries(onnxruntime-jni
        PUBLIC
      ${JNI_LIBRARIES})
  endif()
  target_link_libraries(onnxruntime-jni
      PUBLIC
    ${BEGIN_WHOLE_ARCHIVE}
    onnxruntime_session
    ${onnxruntime_libs}
    ${PROVIDERS_CUDA}
    ${PROVIDERS_MKLDNN}
    ${PROVIDERS_NGRAPH}
    ${PROVIDERS_NNAPI}
    ${PROVIDERS_TENSORRT}
    ${PROVIDERS_OPENVINO}
    onnxruntime_optimizer
    onnxruntime_providers
    onnxruntime_util
    ${onnxruntime_tvm_libs}
    onnxruntime_framework
    ${END_WHOLE_ARCHIVE}
    onnxruntime_graph
    onnxruntime_common
    onnxruntime_mlas
    ${onnxruntime_EXTERNAL_LIBRARIES}
    )
endif()


