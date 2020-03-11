// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/common/logging/sinks/cerr_sink.h"
#include "core/framework/allocatormgr.h"
#include "core/session/environment.h"
#include "core/framework/ml_value.h"
#include "core/session/inference_session.h"

namespace onnxruntime {
namespace python {

namespace py = pybind11;

int OnnxRuntimeTensorToNumpyType(const DataTypeImpl* tensor_type);

MLDataType NumpyTypeToOnnxRuntimeType(int numpy_type);

void CreateGenericMLValue(const onnxruntime::InputDefList* input_def_list, AllocatorPtr alloc, const std::string& name_input,
                          py::object& value, OrtValue* p_mlvalue);
}  // namespace python
}  // namespace onnxruntime
