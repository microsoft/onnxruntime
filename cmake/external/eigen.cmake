
if (onnxruntime_USE_PREINSTALLED_EIGEN)
    add_library(eigen INTERFACE)
    file(TO_CMAKE_PATH ${eigen_SOURCE_PATH} eigen_INCLUDE_DIRS)
    target_include_directories(eigen INTERFACE ${eigen_INCLUDE_DIRS})
else ()
    if(Patch_FOUND)
        set(EIGEN_PATCH_COMMAND ${Patch_EXECUTABLE} --ignore-whitespace -p1 -i
                                ${PROJECT_SOURCE_DIR}/patches/eigen/3.4.0_to_ORT_1.16_src.patch)
    else()
        set(EIGEN_PATCH_COMMAND git apply --ignore-space-change --ignore-whitespace
                                ${PROJECT_SOURCE_DIR}/patches/eigen/3.4.0_to_ORT_1.16_src.patch)
    endif()

    # Why does an ACL have different args which imply `git apply` is intended to be used
    # (which AFAIK doesn't work with the change to download source instead of do a git checkout)
    # https://github.com/microsoft/onnxruntime/issues/15248
    FetchContent_Declare(
        eigen
        URL ${DEP_URL_eigen}
        URL_HASH SHA1=${DEP_SHA1_eigen}
        PATCH_COMMAND ${EIGEN_PATCH_COMMAND}
    )

    FetchContent_Populate(eigen)
    set(eigen_INCLUDE_DIRS  "${eigen_SOURCE_DIR}")
endif()
