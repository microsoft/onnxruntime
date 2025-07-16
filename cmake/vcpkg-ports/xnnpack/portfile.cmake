if(VCPKG_TARGET_IS_WINDOWS)
    vcpkg_check_linkage(ONLY_STATIC_LIBRARY)
endif()

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO google/XNNPACK
    REF 953dcb96cc1b21b4b966952f8ee67a9e1f0d3e71
    SHA512 8c12930ef3b2f832962682d73c362518c014bb4e56d0c5cad2b8b63a03c91dccf6e6a3fd0eb91931fc5872c7df9773e76bf08553fc9c3cc22c94636c74815e94
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
