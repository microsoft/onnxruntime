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
    set(eigen_INCLUDE_DIRS  "${eigen_SOURCE_DIR}")

