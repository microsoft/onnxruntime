vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO microsoft/cpp_client_telemetry
    REF v3.10.173.1
    SHA512 e55bc35274236f57757660073c4dccccab3462342c8566212f1df4bf8824295a2bb3d3d79a11f3950e7c9252641827e9dd3d7c28c421dea3bdaee277e4f2ce32
    HEAD_REF main
    PATCHES
        "${CMAKE_CURRENT_LIST_DIR}/../../patches/cpp_client_telemetry/cpp_client_telemetry.patch"
)

set(BUILD_APPLE_HTTP OFF)
if(VCPKG_TARGET_IS_OSX OR VCPKG_TARGET_IS_IOS)
  set(BUILD_APPLE_HTTP ON)
endif()

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DMATSDK_USE_VCPKG_DEPS=ON
        -DBUILD_HEADERS=ON
        -DBUILD_LIBRARY=ON
        -DBUILD_TEST_TOOL=OFF
        -DBUILD_UNIT_TESTS=OFF
        -DBUILD_FUNC_TESTS=OFF
        -DBUILD_JNI_WRAPPER=OFF
        -DBUILD_OBJC_WRAPPER=OFF
        -DBUILD_SWIFT_WRAPPER=OFF
        -DBUILD_PACKAGE=OFF
        -DBUILD_VERSION=${VERSION}
        -DBUILD_APPLE_HTTP=${BUILD_APPLE_HTTP}
    -DMATSDK_CURL_PROVIDER=SYSTEM
    -DMATSDK_SQLITE_PROVIDER=SYSTEM
    -DMATSDK_ZLIB_PROVIDER=SYSTEM
)

vcpkg_cmake_install()
vcpkg_cmake_config_fixup(PACKAGE_NAME MSTelemetry CONFIG_PATH lib/cmake/MSTelemetry)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
