// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_pybind_exceptions.h"

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
    throw Fail(sType);
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
      default:
        throw std::runtime_error(std::move(msg));
    }
  }
}

}
}