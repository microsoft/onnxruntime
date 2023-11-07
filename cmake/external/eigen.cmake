if (onnxruntime_USE_PREINSTALLED_EIGEN)
    add_library(eigen INTERFACE)
    file(TO_CMAKE_PATH ${eigen_SOURCE_PATH} eigen_INCLUDE_DIRS)
    target_include_directories(eigen INTERFACE ${eigen_INCLUDE_DIRS})
else ()
    if(NOT Patch_FOUND)
        message(FATAL_ERROR "Patch is required but was not found.")
    endif()

    FetchContent_Declare(
        eigen
        URL ${DEP_URL_eigen}
        URL_HASH SHA1=${DEP_SHA1_eigen}
        PATCH_COMMAND ${Patch_EXECUTABLE} --ignore-whitespace -p1 -i
                      ${PROJECT_SOURCE_DIR}/patches/eigen/3.4.0_to_ORT_1.16_src.patch
    )

    FetchContent_Populate(eigen)
    set(eigen_INCLUDE_DIRS  "${eigen_SOURCE_DIR}")
endif()
