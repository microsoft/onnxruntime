vcpkg_buildpath_length_warning(37)

block(SCOPE_FOR VARIABLES PROPAGATE SOURCE_PATH)
set(VCPKG_BUILD_TYPE release) # header-only

vcpkg_from_gitlab(
    GITLAB_URL https://gitlab.com
    OUT_SOURCE_PATH SOURCE_PATH
    REPO libeigen/eigen
    REF 1d8b82b0740839c0de7f1242a3585e3390ff5f33
    SHA512 26a0d86ece9a28c93d64ee3e6ff96276b096a80e34fa2f3c474638824ffa2c56e582d9b2b9ffbb73b18a28b4828b827654b65a3471bc83505c48feab26fb836d
    HEAD_REF master
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DBUILD_TESTING=OFF
        -DEIGEN_BUILD_DOC=OFF
        -DEIGEN_BUILD_BLAS=OFF
        -DEIGEN_BUILD_LAPACK=OFF
        -DEIGEN_BUILD_PKGCONFIG=ON
        "-DCMAKEPACKAGE_INSTALL_DIR=${CURRENT_PACKAGES_DIR}/share/eigen3"
        "-DPKGCONFIG_INSTALL_DIR=${CURRENT_PACKAGES_DIR}/lib/pkgconfig"
)

vcpkg_cmake_install()
vcpkg_cmake_config_fixup()
vcpkg_replace_string("${CURRENT_PACKAGES_DIR}/share/eigen3/Eigen3Config.cmake" "if (NOT TARGET eigen)" "if (NOT TARGET Eigen3::Eigen)")
endblock()

if(NOT VCPKG_BUILD_TYPE)
    file(INSTALL "${CURRENT_PACKAGES_DIR}/lib/pkgconfig/eigen3.pc" DESTINATION "${CURRENT_PACKAGES_DIR}/debug/lib/pkgconfig")
endif()
vcpkg_fixup_pkgconfig()

file(GLOB INCLUDES "${CURRENT_PACKAGES_DIR}/include/eigen3/*")
# Copy the eigen header files to conventional location for user-wide MSBuild integration
file(COPY ${INCLUDES} DESTINATION "${CURRENT_PACKAGES_DIR}/include")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/COPYING.README")
