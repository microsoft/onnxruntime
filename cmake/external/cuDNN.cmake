add_library(CUDNN::cudnn_all INTERFACE IMPORTED)

find_path(
    CUDNN_INCLUDE_DIR cudnn.h
    HINTS $ENV{CUDNN_PATH} ${CUDNN_PATH} ${Python_SITEARCH}/nvidia/cudnn ${CUDAToolkit_INCLUDE_DIRS}
    PATH_SUFFIXES include
    REQUIRED
)

file(READ "${CUDNN_INCLUDE_DIR}/cudnn_version.h" cudnn_version_header)
string(REGEX MATCH "#define CUDNN_MAJOR [1-9]+" macrodef "${cudnn_version_header}")
string(REGEX MATCH "[1-9]+" CUDNN_MAJOR_VERSION "${macrodef}")

function(find_cudnn_library NAME)
    find_library(
        ${NAME}_LIBRARY ${NAME} "lib${NAME}.so.${CUDNN_MAJOR_VERSION}"
        HINTS $ENV{CUDNN_PATH} ${CUDNN_PATH} ${Python_SITEARCH}/nvidia/cudnn ${CUDAToolkit_LIBRARY_DIR}
        PATH_SUFFIXES lib64 lib/x64 lib
        REQUIRED
    )
    
    if(${NAME}_LIBRARY)
        add_library(CUDNN::${NAME} UNKNOWN IMPORTED)
        set_target_properties(
            CUDNN::${NAME} PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES ${CUDNN_INCLUDE_DIR}
            IMPORTED_LOCATION ${${NAME}_LIBRARY}
        )
        message(STATUS "${NAME} found at ${${NAME}_LIBRARY}.")
    else()
        message(STATUS "${NAME} not found.")
    endif()


endfunction()

find_cudnn_library(cudnn)

include (FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    LIBRARY REQUIRED_VARS
    CUDNN_INCLUDE_DIR cudnn_LIBRARY
)

if(CUDNN_INCLUDE_DIR AND cudnn_LIBRARY)

    message(STATUS "cuDNN: ${cudnn_LIBRARY}")
    message(STATUS "cuDNN: ${CUDNN_INCLUDE_DIR}")
    
    set(CUDNN_FOUND ON CACHE INTERNAL "cuDNN Library Found")

else()

    set(CUDNN_FOUND OFF CACHE INTERNAL "cuDNN Library Not Found")

endif()

target_include_directories(
    CUDNN::cudnn_all
    INTERFACE
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CUDNN_INCLUDE_DIR}>
)

target_link_libraries(
    CUDNN::cudnn_all
    INTERFACE
    CUDNN::cudnn 
)

if(CUDNN_MAJOR_VERSION EQUAL 8)
    find_cudnn_library(cudnn_adv_infer)
    find_cudnn_library(cudnn_adv_train)
    find_cudnn_library(cudnn_cnn_infer)
    find_cudnn_library(cudnn_cnn_train)
    find_cudnn_library(cudnn_ops_infer)
    find_cudnn_library(cudnn_ops_train)

    target_link_libraries(
        CUDNN::cudnn_all
        INTERFACE
        CUDNN::cudnn_adv_train
        CUDNN::cudnn_ops_train
        CUDNN::cudnn_cnn_train
        CUDNN::cudnn_adv_infer
        CUDNN::cudnn_cnn_infer
        CUDNN::cudnn_ops_infer
    )
elseif(CUDNN_MAJOR_VERSION EQUAL 9)
    find_cudnn_library(cudnn_cnn)
    find_cudnn_library(cudnn_adv)
    find_cudnn_library(cudnn_graph)
    find_cudnn_library(cudnn_ops)
    find_cudnn_library(cudnn_engines_runtime_compiled)
    find_cudnn_library(cudnn_engines_precompiled)
    find_cudnn_library(cudnn_heuristic)

    target_link_libraries(
        CUDNN::cudnn_all
        INTERFACE
        CUDNN::cudnn_adv
        CUDNN::cudnn_ops
        CUDNN::cudnn_cnn
        CUDNN::cudnn_graph
        CUDNN::cudnn_engines_runtime_compiled
        CUDNN::cudnn_engines_precompiled
        CUDNN::cudnn_heuristic
    )
endif()
