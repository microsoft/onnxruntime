

# The libjpeg-turbo owner says ExternalProject is supported and FetchContent will never be.

if (NOT OCOS_BUILD_ANDROID)
    ExternalProject_Add(libjpeg-turbo
        GIT_REPOSITORY  https://github.com/libjpeg-turbo/libjpeg-turbo.git
        GIT_TAG         2.1.4
        GIT_SHALLOW     TRUE
        PREFIX          ${CMAKE_CURRENT_BINARY_DIR}/_deps/libjpeg-turbo
        INSTALL_COMMAND cmake -E echo "Skipping install step for dependency libjpeg-turbo"
        INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}
    )
else()
    message(STATUS "Toolchain: ${CMAKE_TOOLCHAIN_FILE}")

    ExternalProject_Add(libjpeg-turbo
        GIT_REPOSITORY  https://github.com/libjpeg-turbo/libjpeg-turbo.git
        GIT_TAG         2.1.4
        GIT_SHALLOW     TRUE
        PREFIX          ${CMAKE_CURRENT_BINARY_DIR}/_deps/libjpeg-turbo
        INSTALL_COMMAND cmake -E echo "Skipping install step for dependency libjpeg-turbo"
        INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}
        CMAKE_ARGS
            -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
            -DANDROID_ABI=${ANDROID_ABI}
            -DANDROID_PLATFORM=${ANDROID_PLATFORM}
    )
endif()

set (libjpeg-turbo_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps/libjpeg-turbo/src/libjpeg-turbo)
set (libjpeg-turbo_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps/libjpeg-turbo/src/libjpeg-turbo-build)
set (libjpeg-turbo_INCLUDE_DIRS ${libjpeg-turbo_SOURCE_DIR} ${libjpeg-turbo_BINARY_DIR})

# adjust for diffs in output name and location
if (MSVC)
    link_directories(${libjpeg-turbo_BINARY_DIR}/${CMAKE_BUILD_TYPE})
    set(libjpeg-turbo_LIB_NAME jpeg-static)
else()
    link_directories(${libjpeg-turbo_BINARY_DIR})
    set(libjpeg-turbo_LIB_NAME jpeg)
endif()
