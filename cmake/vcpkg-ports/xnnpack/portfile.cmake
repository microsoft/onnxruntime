if(VCPKG_TARGET_IS_WINDOWS)
    vcpkg_check_linkage(ONLY_STATIC_LIBRARY)
endif()

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO google/XNNPACK
    REF 3cf85e705098622d59056dcb8f5f963ea7bb0a00
    SHA512 af10afde80def08dc3b20a35bd38e84f9f749865ecc4bc9733b5d99d8a2f0f30c19c3f23472d65462a907b3a58226e3b254354a92a6baa31031824f68012a055
    HEAD_REF master
    PATCHES
        fix-build.patch
        disable_gcc_warning.patch
)
vcpkg_find_acquire_program(PYTHON3)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    WINDOWS_USE_MSBUILD
    OPTIONS
        "-DPython3_EXECUTABLE=${PYTHON3}"
        "-DPython_EXECUTABLE=${PYTHON3}"
        -DXNNPACK_USE_SYSTEM_LIBS=ON
        -DXNNPACK_ENABLE_AVXVNNI=OFF
        -DXNNPACK_ENABLE_ASSEMBLY=ON
        -DXNNPACK_ENABLE_MEMOPT=ON
        -DXNNPACK_ENABLE_SPARSE=ON
        -DXNNPACK_ENABLE_KLEIDIAI=OFF
        -DXNNPACK_BUILD_TESTS=OFF
        -DXNNPACK_BUILD_BENCHMARKS=OFF
)
vcpkg_cmake_install()
vcpkg_copy_pdbs()

file(INSTALL "${SOURCE_PATH}/LICENSE" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}" RENAME copyright)
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include"
                    "${CURRENT_PACKAGES_DIR}/debug/bin"
                    "${CURRENT_PACKAGES_DIR}/debug/share"
)
