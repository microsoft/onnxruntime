if (onnxruntime_USE_PREINSTALLED_EIGEN)
    add_library(eigen INTERFACE)
    file(TO_CMAKE_PATH ${eigen_SOURCE_PATH} eigen_INCLUDE_DIRS)
    target_include_directories(eigen INTERFACE ${eigen_INCLUDE_DIRS})
else ()
    FetchContent_Declare(
        eigen
        URL ${DEP_URL_eigen}
        URL_HASH SHA1=${DEP_SHA1_eigen}
    )

    FetchContent_Populate(eigen)
    set(eigen_INCLUDE_DIRS  "${eigen_SOURCE_DIR}")
endif()
