if (onnxruntime_USE_PREINSTALLED_EIGEN)
    add_library(eigen INTERFACE)
    file(TO_CMAKE_PATH ${eigen_SOURCE_PATH} eigen_INCLUDE_DIRS)
    target_include_directories(eigen INTERFACE ${eigen_INCLUDE_DIRS})
else ()
    if(CMAKE_SYSTEM_NAME MATCHES "AIX")
        FetchContent_Declare(
            eigen
            URL ${DEP_URL_eigen}
            URL_HASH SHA1=${DEP_SHA1_eigen}
            PATCH_COMMAND ${Patch_EXECUTABLE} --binary --ignore-whitespace -p1 --input=${PROJECT_SOURCE_DIR}/patches/eigen/eigen-aix.patch
            SOURCE_DIR ${BUILD_DIR_NO_CONFIG}/_deps/eigen-src
            BINARY_DIR ${CMAKE_BINARY_DIR}/deps/eigen-build
            DOWNLOAD_DIR ${BUILD_DIR_NO_CONFIG}/_deps/eigen-download
        )
    else()
        FetchContent_Declare(
            eigen
            URL ${DEP_URL_eigen}
            URL_HASH SHA1=${DEP_SHA1_eigen}
            SOURCE_DIR ${BUILD_DIR_NO_CONFIG}/_deps/eigen-src
            BINARY_DIR ${CMAKE_BINARY_DIR}/deps/eigen-build
            DOWNLOAD_DIR ${BUILD_DIR_NO_CONFIG}/_deps/eigen-download
        )
    endif()

    FetchContent_Populate(eigen)
    set(eigen_INCLUDE_DIRS  "${eigen_SOURCE_DIR}")
endif()
