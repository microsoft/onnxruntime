if(NOT TARGET Eigen3::Eigen)
    if(CMAKE_SYSTEM_NAME MATCHES "AIX")
        onnxruntime_fetchcontent_declare(
            eigen
            URL ${DEP_URL_eigen}
            URL_HASH SHA1=${DEP_SHA1_eigen}
            PATCH_COMMAND ${Patch_EXECUTABLE} --binary --ignore-whitespace -p1 < ${PROJECT_SOURCE_DIR}/patches/eigen/eigen-aix.patch
	    EXCLUDE_FROM_ALL
        )
    else()
        onnxruntime_fetchcontent_declare(
            eigen
            URL ${DEP_URL_eigen}
            URL_HASH SHA1=${DEP_SHA1_eigen}
            PATCH_COMMAND ${Patch_EXECUTABLE} --binary --ignore-whitespace -p1 < ${PROJECT_SOURCE_DIR}/patches/eigen/eigen-edge.patch
	    EXCLUDE_FROM_ALL
        )
    endif()

    FetchContent_Populate(eigen)
    add_library(Eigen3::Eigen INTERFACE)
    target_include_directories(Eigen3::Eigen INTERFACE ${eigen_SOURCE_DIR})
endif()
