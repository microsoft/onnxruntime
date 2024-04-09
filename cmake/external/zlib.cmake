if (NOT MSVC)
    # Assume zlib is found using find_package. TODO: Could fallback to FetchContent if find_package fails.
    find_package(ZLIB REQUIRED)
    set(zlib_INCLUDE_DIRS ${ZLIB_INCLUDE_DIRS})
    set(zlib_LIB_NAME ${ZLIB_LIBRARIES})
else()
    FetchContent_Declare(
        zlib
        GIT_REPOSITORY  "https://github.com/madler/zlib.git"
        GIT_TAG         v1.2.13
        GIT_SHALLOW     TRUE
    )

    FetchContent_MakeAvailable(zlib)
    set(zlib_INCLUDE_DIRS ${zlib_SOURCE_DIR} ${zlib_BINARY_DIR})
    set(ZLIB_LIBRARIES zlibstatic)
endif()
