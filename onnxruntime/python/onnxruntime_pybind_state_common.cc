#include "onnxruntime_pybind_exceptions.h"
#include "onnxruntime_pybind_state_common.h"

const std::string onnxruntime::python::SessionObjectInitializer::default_logger_id = "Default";

namespace onnxruntime {
namespace python {
namespace py = pybind11;

void ThrowIfPyErrOccured() {
  if (PyErr_Occurred()) {
    PyObject *ptype, *pvalue, *ptraceback;
    PyErr_Fetch(&ptype, &pvalue, &ptraceback);

    PyObject* pStr = PyObject_Str(ptype);
    std::string sType = py::reinterpret_borrow<py::str>(pStr);
    Py_XDECREF(pStr);
    pStr = PyObject_Str(pvalue);
    sType += ": ";
    sType += py::reinterpret_borrow<py::str>(pStr);
    Py_XDECREF(pStr);
    throw std::runtime_error(sType);
  }
}

}  // namespace python
}  // namespace onnxruntime