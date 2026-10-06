add_library(CUDNN::cudnn_all INTERFACE IMPORTED)

find_path(
    CUDNN_INCLUDE_DIR cudnn.h
    HINTS $ENV{CUDNN_PATH} ${CUDNN_PATH} ${Python_SITEARCH}/nvidia/cudnn ${CUDAToolkit_INCLUDE_DIRS}
    PATH_SUFFIXES include include/${onnxruntime_CUDA_VERSION}
    REQUIRED
)

file(READ "${CUDNN_INCLUDE_DIR}/cudnn_version.h" cudnn_version_header)
string(REGEX MATCH "#define CUDNN_MAJOR [1-9]+" macrodef "${cudnn_version_header}")
string(REGEX MATCH "[1-9]+" CUDNN_MAJOR_VERSION "${macrodef}")

function(find_cudnn_library NAME)
    if(WIN32)
        # Since CUDA 13.x, Windows cuDNN import libraries live under an architecture-specific
        # subdirectory (lib/x64 for win-x64, lib/arm64 for win-arm64). A single cuDNN package
        # may ship both arches, so only search the subdirectory matching the current target
        # platform — otherwise find_library could pick the wrong arch (it returns the first
        # match in PATH_SUFFIXES order).
        if(onnxruntime_target_platform STREQUAL "ARM64" OR onnxruntime_target_platform STREQUAL "ARM64EC")
            set(_cudnn_arch_suffixes lib/arm64 lib/${onnxruntime_CUDA_VERSION}/arm64)
        else()
            set(_cudnn_arch_suffixes lib/x64 lib/${onnxruntime_CUDA_VERSION}/x64)
        endif()
    endif()
    find_library(
        ${NAME}_LIBRARY ${NAME} "lib${NAME}.so.${CUDNN_MAJOR_VERSION}"
        HINTS $ENV{CUDNN_PATH} ${CUDNN_PATH} ${Python_SITEARCH}/nvidia/cudnn ${CUDAToolkit_LIBRARY_DIR}
        PATH_SUFFIXES lib64 ${_cudnn_arch_suffixes} lib
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
