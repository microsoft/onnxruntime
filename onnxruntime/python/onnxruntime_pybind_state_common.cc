#include "onnxruntime_pybind_exceptions.h"
#include "onnxruntime_pybind_state_common.h"

namespace onnxruntime {
namespace python {
namespace py = pybind11;

const std::string onnxruntime::python::SessionObjectInitializer::default_logger_id = "Default";

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
    throw Fail(sType);
  }
}

#ifdef ENABLE_TRAINING

void DlpackCapsuleDestructor(PyObject* data) {
  DLManagedTensor* dlmanged_tensor = (DLManagedTensor*)PyCapsule_GetPointer(data, "dltensor");
  if (dlmanged_tensor) {
    // the dlmanged_tensor has not been consumed, call deleter ourselves.
    dlmanged_tensor->deleter(const_cast<DLManagedTensor*>(dlmanged_tensor));
  } else {
    // the dlmanged_tensor has been consumed,
    // PyCapsule_GetPointer has set an error indicator.
    PyErr_Clear();
  }
}

py::object ToDlpack(OrtValue& ort_value) {
  DLManagedTensor* dlmanaged_tensor = dlpack::OrtValueToDlpack(ort_value);
  return py::reinterpret_steal<py::object>(
      PyCapsule_New(dlmanaged_tensor, "dltensor", DlpackCapsuleDestructor));
}

OrtValue FromDlpack(py::object dlpack_tensor, const bool is_bool_tensor) {
  DLManagedTensor* dlmanaged_tensor = (DLManagedTensor*)PyCapsule_GetPointer(dlpack_tensor.ptr(), "dltensor");
  OrtValue ort_value = dlpack::DlpackToOrtValue(dlmanaged_tensor, is_bool_tensor);
  // Make sure this capsule will never be used again.
  PyCapsule_SetName(dlpack_tensor.ptr(), "used_dltensor");
  return ort_value;
}

#endif

}  // namespace python
}  // namespace onnxruntime
