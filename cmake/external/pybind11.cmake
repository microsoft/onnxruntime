onnxruntime_fetchcontent_declare(
    pybind11_project
    URL ${DEP_URL_pybind11}
    URL_HASH SHA1=${DEP_SHA1_pybind11}
    EXCLUDE_FROM_ALL
    FIND_PACKAGE_ARGS 2.6 NAMES pybind11
)
onnxruntime_fetchcontent_makeavailable(pybind11_project)

if(TARGET pybind11::module)
  set(pybind11_lib pybind11::module)
else()
  set(pybind11_dep pybind11::pybind11)
endif()
