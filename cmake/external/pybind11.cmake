set(PYBIND11_NOPYTHON ON CACHE BOOL "" FORCE)
set(PYBIND11_INSTALL OFF CACHE BOOL "" FORCE)
set(PYBIND11_TEST OFF CACHE BOOL "" FORCE)
onnxruntime_fetchcontent_declare(
    pybind11_project
    URL ${DEP_URL_pybind11}
    URL_HASH SHA1=${DEP_SHA1_pybind11}
    EXCLUDE_FROM_ALL
    FIND_PACKAGE_ARGS 2.13 NAMES pybind11
)
onnxruntime_fetchcontent_makeavailable(pybind11_project)

