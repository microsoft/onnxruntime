include (ExternalProject)

if (onnxruntime_USE_PREINSTALLED_EIGEN)
    file(TO_CMAKE_PATH ${eigen_SOURCE_PATH} eigen_INCLUDE_DIRS)
    ExternalProject_Add(eigen
        PREFIX eigen
        SOURCE_DIR ${eigen_SOURCE_PATH}
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        DOWNLOAD_COMMAND ""
        UPDATE_COMMAND ""
    )
else ()
    set(eigen_URL "https://github.com/eigenteam/eigen-git-mirror.git")
    set(eigen_TAG "3.3.7")
    set(eigen_ROOT_DIR ${CMAKE_CURRENT_BINARY_DIR}/external/eigen)
    set(eigen_INCLUDE_DIRS ${eigen_ROOT_DIR})
    ExternalProject_Add(eigen
        PREFIX eigen
        GIT_REPOSITORY ${eigen_URL}
        GIT_TAG ${eigen_TAG}
        SOURCE_DIR ${eigen_ROOT_DIR}
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
    )
endif()
