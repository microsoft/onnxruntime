if(VCPKG_TARGET_IS_WINDOWS)
    vcpkg_check_linkage(ONLY_STATIC_LIBRARY)
endif()

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO google/XNNPACK
    REF 5ede268353f330f8b7385ce812957804dfe56685 # 2025-01-10
    SHA512 6610bd90fca6cbd1ebc95ce0c6d17afc1e3d8100f2630d87f90e7520e74af87b2b00e2237fdaf069d45269daadbb1fb64358ed6d5be582ee7132fb6bd54a03fa
    HEAD_REF master
    PATCHES
        fix-build.patch
)
vcpkg_find_acquire_program(PYTHON3)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
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
