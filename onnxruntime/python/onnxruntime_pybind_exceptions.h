// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

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

void RegisterExceptions(pybind11::module& m);

void OrtPybindThrowIfError(onnxruntime::common::Status status);

}  // namespace python
}  // namespace onnxruntime
