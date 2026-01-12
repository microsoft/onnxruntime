# On Windows, we can get a cpuinfo.dll, but it exports no symbols.
if(VCPKG_TARGET_IS_WINDOWS)
    vcpkg_check_linkage(ONLY_STATIC_LIBRARY)
endif()

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO pytorch/cpuinfo
    REF 403d652dca4c1046e8145950b1c0997a9f748b57
    SHA512 f7cd6dc44bd1120af610cae1337ed4c0f557ba78d2de9c73fed350fa3dfe9512643a1619ae55f5a540c6316a87d641856cca27297bb8766e48f39b7b7a59da1f
    HEAD_REF master
    PATCHES
        patch_cpuinfo_h_for_arm64ec.patch
        patch_vcpkg_arm64ec_support.patch       # https://github.com/pytorch/cpuinfo/pull/324
        win_arm_fp16_detection_fallback.patch   # https://github.com/pytorch/cpuinfo/pull/348
)

vcpkg_check_features(OUT_FEATURE_OPTIONS FEATURE_OPTIONS
    FEATURES
        tools CPUINFO_BUILD_TOOLS
)

set(LINK_OPTIONS "")
if(VCPKG_LIBRARY_LINKAGE STREQUAL "dynamic")
    list(APPEND LINK_OPTIONS -DCPUINFO_LIBRARY_TYPE=shared)
else()
    list(APPEND LINK_OPTIONS -DCPUINFO_LIBRARY_TYPE=static)
endif()

if(VCPKG_CRT_LINKAGE STREQUAL "dynamic")
    list(APPEND LINK_OPTIONS -DCPUINFO_RUNTIME_TYPE=shared)
else()
    list(APPEND LINK_OPTIONS -DCPUINFO_RUNTIME_TYPE=static)
endif()

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        ${FEATURE_OPTIONS}
        ${LINK_OPTIONS}
        -DCPUINFO_BUILD_UNIT_TESTS=OFF
        -DCPUINFO_BUILD_MOCK_TESTS=OFF
        -DCPUINFO_BUILD_BENCHMARKS=OFF
    OPTIONS_DEBUG
        -DCPUINFO_LOG_LEVEL=debug
    OPTIONS_RELEASE
        -DCPUINFO_LOG_LEVEL=default
)
vcpkg_cmake_install()
vcpkg_cmake_config_fixup()
vcpkg_copy_pdbs()
vcpkg_fixup_pkgconfig() # pkg_check_modules(libcpuinfo)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")

if("tools" IN_LIST FEATURES)
    set(additional_tools "")
    if(EXISTS "${CURRENT_PACKAGES_DIR}/bin/cpuid-dump${VCPKG_TARGET_EXECUTABLE_SUFFIX}")
        list(APPEND additional_tools "cpuid-dump")
    endif()
    vcpkg_copy_tools(
        TOOL_NAMES cache-info cpu-info isa-info ${additional_tools}
        AUTO_CLEAN
    )
endif()

file(INSTALL "${SOURCE_PATH}/LICENSE" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}" RENAME copyright)
