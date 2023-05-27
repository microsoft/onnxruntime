
if (onnxruntime_USE_PREINSTALLED_EIGEN)
    add_library(eigen INTERFACE)
    file(TO_CMAKE_PATH ${eigen_SOURCE_PATH} eigen_INCLUDE_DIRS)
    target_include_directories(eigen INTERFACE ${eigen_INCLUDE_DIRS})
else ()
    if (onnxruntime_USE_ACL)
        FetchContent_Declare(
        eigen
        URL ${DEP_URL_eigen}
        URL_HASH SHA1=${DEP_SHA1_eigen}
        PATCH_COMMAND ${Patch_EXECUTABLE} --ignore-space-change --ignore-whitespace < ${PROJECT_SOURCE_DIR}/patches/eigen/Fix_Eigen_Build_Break.patch
		)
    else()
        FetchContent_Declare(
        eigen
        URL ${DEP_URL_eigen}
        URL_HASH SHA1=${DEP_SHA1_eigen}
		)
    endif()
    FetchContent_Populate(eigen)
    set(eigen_INCLUDE_DIRS  "${eigen_SOURCE_DIR}")
endif()
