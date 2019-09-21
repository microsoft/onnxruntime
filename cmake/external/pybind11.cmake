include(ExternalProject)

set(pybind11_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/pybind11/src/pybind11/include)
set(pybind11_URL https://github.com/pybind/pybind11.git)
set(pybind11_TAG v2.4.0)

ExternalProject_Add(pybind11
        PREFIX pybind11
        GIT_REPOSITORY ${pybind11_URL}
        GIT_TAG ${pybind11_TAG}
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
)
