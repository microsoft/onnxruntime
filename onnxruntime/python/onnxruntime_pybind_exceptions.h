// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <pybind11/pybind11.h>
#include <stdexcept>
#include "core/common/status.h"

namespace onnxruntime {
namespace python {

// onnxruntime::python exceptions map 1:1 to onnxruntime:common::StatusCode enum.
struct Fail : std::runtime_error {
  explicit Fail(const std::string& what) : std::runtime_error(what) {}
};
struct InvalidArgument : std::runtime_error {
  explicit InvalidArgument(const std::string& what) : std::runtime_error(what) {}
};
struct NoSuchFile : std::runtime_error {
  explicit NoSuchFile(const std::string& what) : std::runtime_error(what) {}
};
struct NoModel : std::runtime_error {
  explicit NoModel(const std::string& what) : std::runtime_error(what) {}
};
struct EngineError : std::runtime_error {
  explicit EngineError(const std::string& what) : std::runtime_error(what) {}
};
struct RuntimeException : std::runtime_error {
  explicit RuntimeException(const std::string& what) : std::runtime_error(what) {}
};
struct InvalidProtobuf : std::runtime_error {
  explicit InvalidProtobuf(const std::string& what) : std::runtime_error(what) {}
};
struct ModelLoaded : std::runtime_error {
  explicit ModelLoaded(const std::string& what) : std::runtime_error(what) {}
};
struct NotImplemented : std::runtime_error {
  explicit NotImplemented(const std::string& what) : std::runtime_error(what) {}
};
struct InvalidGraph : std::runtime_error {
  explicit InvalidGraph(const std::string& what) : std::runtime_error(what) {}
};
struct EPFail : std::runtime_error {
  explicit EPFail(const std::string& what) : std::runtime_error(what) {}
};

inline void RegisterExceptions(pybind11::module& m) {
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

inline void OrtPybindThrowIfError(onnxruntime::common::Status status) {
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
}  // namespace python
}  // namespace onnxruntime
