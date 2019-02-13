include (ExternalProject)

set(OPENVINO_URL https://github.com/opencv/dldt.git)
set(OPENVINO_TAG 2018)

set(OPENVINO_SHARED_LIB libinference_engine.so)

if(onnxruntime_USE_OPENVINO)

    set(OPENVINO_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/openvino/src/dldt)
    set(OPENVINO_INSTALL ${CMAKE_CURRENT_BINARY_DIR}/openvino/install)
    set(OPENVINO_LIB_DIR ${OPENVINO_SOURCE}/inference-engine/bin/intel64/Release/lib)

    set(OPENVINO_INCLUDE_DIR ${OPENVINO_SOURCE}/inference-engine/include)


    ExternalProject_Add(project_openvino
        PREFIX openvino
        GIT_REPOSITORY ${OPENVINO_URL}
        GIT_TAG ${OPENVINO_TAG}
        SOURCE_DIR ${OPENVINO_SOURCE}
        SOURCE_SUBDIR inference-engine
        CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${OPENVINO_INSTALL}
    )
    link_directories(${OPENVINO_LIB_DIR})
endif()