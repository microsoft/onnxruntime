// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_pybind_exceptions.h"

namespace onnxruntime {
namespace python {
namespace py = pybind11;

void ThrowIfPyErrOccured() {
  if (PyErr_Occurred()) {
    // Enhanced Python 3.14+ compatible exception handling
    PyObject *ptype, *pvalue, *ptraceback;
    PyErr_Fetch(&ptype, &pvalue, &ptraceback);
    
    // Normalize the exception (important for Python 3.14+ compatibility)
    PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);

    std::string error_message;
    
    try {
      // Safe string extraction with proper error handling
      if (ptype != nullptr) {
        PyObject* ptype_str = PyObject_Str(ptype);
        if (ptype_str != nullptr) {
          try {
            error_message += py::reinterpret_borrow<py::str>(ptype_str);
          } catch (const py::error_already_set&) {
            error_message += "<type conversion failed>";
          }
          Py_DECREF(ptype_str);
        } else {
          error_message += "<unknown type>";
        }
      }

      error_message += ": ";

      if (pvalue != nullptr) {
        PyObject* pvalue_str = PyObject_Str(pvalue);
        if (pvalue_str != nullptr) {
          try {
            error_message += py::reinterpret_borrow<py::str>(pvalue_str);
          } catch (const py::error_already_set&) {
            error_message += "<value conversion failed>";
          }
          Py_DECREF(pvalue_str);
        } else {
          error_message += "<unknown value>";
        }
      } else {
        error_message += "<no error message>";
      }

    } catch (...) {
      // Fallback for any unexpected errors during string conversion
      error_message = "Python exception occurred but details could not be extracted";
    }

    // Clean up references safely
    Py_XDECREF(ptype);
    Py_XDECREF(pvalue);
    Py_XDECREF(ptraceback);

    // Clear any remaining error state
    PyErr_Clear();
    
    throw Fail(std::move(error_message));
  }
}

void RegisterExceptions(pybind11::module& m) {
  pybind11::register_exception<Fail>(m, "Fail");
  pybind11::register_exception<InvalidArgument>(m, "InvalidArgument");
  pybind11::register_exception<NoSuchFile>(m, "NoSuchFile");
  pybind11::register_exception<NoModel>(m, "NoModel");
  pybind11::register_exception<EngineError>(m, "EngineError");
  pybind11::register_exception<RuntimeException>(m, "RuntimeException");
  pybind11::register_exception<InvalidProtobuf>(m, "InvalidProtobuf");
  pybind11::register_exception<ModelLoaded>(m, "ModelLoaded");
  pybind11::register_exception<NotImplemented>(m, "NotImplemented");
  pybind11::register_exception<InvalidGraph>(m, "InvalidGraph");
  pybind11::register_exception<EPFail>(m, "EPFail");
  pybind11::register_exception<ModelLoadCanceled>(m, "ModelLoadCanceled");
  pybind11::register_exception<ModelRequiresCompilation>(m, "ModelRequiresCompilation");
  pybind11::register_exception<NotFound>(m, "NotFound");
}

void OrtPybindThrowIfError(onnxruntime::common::Status status) {
  std::string msg = status.ToString();
  if (!status.IsOK()) {
    switch (status.Code()) {
      case onnxruntime::common::StatusCode::FAIL:
        throw Fail(std::move(msg));
      case onnxruntime::common::StatusCode::INVALID_ARGUMENT:
        throw InvalidArgument(std::move(msg));
      case onnxruntime::common::StatusCode::NO_SUCHFILE:
        throw NoSuchFile(std::move(msg));
      case onnxruntime::common::StatusCode::NO_MODEL:
        throw NoModel(std::move(msg));
      case onnxruntime::common::StatusCode::ENGINE_ERROR:
        throw EngineError(std::move(msg));
      case onnxruntime::common::StatusCode::RUNTIME_EXCEPTION:
        throw RuntimeException(std::move(msg));
      case onnxruntime::common::StatusCode::INVALID_PROTOBUF:
        throw InvalidProtobuf(std::move(msg));
      case onnxruntime::common::StatusCode::NOT_IMPLEMENTED:
        throw NotImplemented(std::move(msg));
      case onnxruntime::common::StatusCode::INVALID_GRAPH:
        throw InvalidGraph(std::move(msg));
      case onnxruntime::common::StatusCode::EP_FAIL:
        throw EPFail(std::move(msg));
      case onnxruntime::common::StatusCode::MODEL_LOAD_CANCELED:
        throw ModelLoadCanceled(std::move(msg));
      case onnxruntime::common::StatusCode::MODEL_REQUIRES_COMPILATION:
        throw ModelRequiresCompilation(std::move(msg));
      case onnxruntime::common::StatusCode::NOT_FOUND:
        throw NotFound(std::move(msg));
      default:
        throw std::runtime_error(std::move(msg));
    }
  }
}

}  // namespace python
}  // namespace onnxruntime