// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#ifdef  _WIN32
#pragma warning(disable : 4505) // warning C4505: '_import_array': unreferenced local function has been removed. _import_array is definded in 'numpy/arrayobject.h'.
#endif
#include <numpy/arrayobject.h>

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

void CreateGenericMLValue(const onnxruntime::InputDefList* input_def_list, const AllocatorPtr& alloc, const std::string& name_input,
                          py::object& value, OrtValue* p_mlvalue);

void CreateTensorMLValue(const AllocatorPtr& alloc, const std::string& name_input, PyArrayObject* pyObject,
                         OrtValue* p_mlvalue);

template <class T>
struct DecRefFn {
  void operator()(T* pyobject) const {
    Py_XDECREF(pyobject);
  }
};

template <class T>
using UniqueDecRefPtr = std::unique_ptr<T, DecRefFn<T>>;

}  // namespace python
}  // namespace onnxruntime
