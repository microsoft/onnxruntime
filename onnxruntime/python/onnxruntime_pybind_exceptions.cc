// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_pybind_exceptions.h"

namespace onnxruntime {
namespace python {

/**
 * Mainly it is used with the `CreateGenericMLValue` function.
 * The function ThrowIfPyErrOccured() is a bridge between the low-level Python C-API and the C++ exception handling
 * used in the onnx runtime project. Its purpose is to be called after a CPython API function that might fail (e.g., PyArray_FromAny). 
 * If that C-API function fails, it doesn't throw a C++ exception; instead, it sets a global error indicator within the Python interpreter.
 */
void ThrowIfPyErrOccured() {
  if (PyErr_Occurred()) {
    throw nanobind::python_error();
  }
}

void RegisterExceptions(nanobind::module_& m) {
  nanobind::exception<Fail>(m, "Fail");
  nanobind::exception<InvalidArgument>(m, "InvalidArgument");
  nanobind::exception<NoSuchFile>(m, "NoSuchFile");
  nanobind::exception<NoModel>(m, "NoModel");
  nanobind::exception<EngineError>(m, "EngineError");
  nanobind::exception<RuntimeException>(m, "RuntimeException");
  nanobind::exception<InvalidProtobuf>(m, "InvalidProtobuf");
  nanobind::exception<ModelLoaded>(m, "ModelLoaded");
  nanobind::exception<NotImplemented>(m, "NotImplemented");
  nanobind::exception<InvalidGraph>(m, "InvalidGraph");
  nanobind::exception<EPFail>(m, "EPFail");
  nanobind::exception<ModelLoadCanceled>(m, "ModelLoadCanceled");
  nanobind::exception<ModelRequiresCompilation>(m, "ModelRequiresCompilation");
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
      default:
        throw std::runtime_error(std::move(msg));
    }
  }
}

}  // namespace python
}  // namespace onnxruntime