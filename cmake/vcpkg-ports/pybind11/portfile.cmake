# Manually define the download for the .zip archive (to be consistent with deps.txt)
# If we used vcpkg_from_github, it would download the .tar.gz archive,
# which has different SHA512: 19bee2c76320e25202ee078b5680ff8a7acfb33494dec29dad984ab04de8bcb01340d9fec37c8cc5ac9015dfc367e60312dcd8506e66ce8f0af4c49db562ddef
vcpkg_download_distfile(ARCHIVE
    URLS "https://github.com/pybind/pybind11/archive/refs/tags/v${VERSION}.zip"
    FILENAME "pybind11-${VERSION}.zip"
    SHA512 786b1bf534ac67a8d5669f8babf67bb13e48b3a3da1b6344e43ae10a84b80bbc8fea5f12a65fd18739c341fefef5622c5dc096db964dff33cc62ea4259b2e2c1
)

vcpkg_extract_source_archive(
    SOURCE_PATH
    ARCHIVE "${ARCHIVE}"
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DPYBIND11_TEST=OFF
        # Disable all Python searching, Python required only for tests
        -DPYBIND11_NOPYTHON=ON
)

vcpkg_cmake_install()
vcpkg_cmake_config_fixup(CONFIG_PATH "share/cmake/pybind11")
vcpkg_fixup_pkgconfig()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/")

file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/usage" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
