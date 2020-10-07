// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

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

bool IsNumericNumpyType(int npy_type);

bool IsNumericNumpyArray(py::object& py_object);

int OnnxRuntimeTensorToNumpyType(const DataTypeImpl* tensor_type);

MLDataType NumpyTypeToOnnxRuntimeType(int numpy_type);

using MemCpyFunc = void (*)(void*, const void*, size_t);
void CpuToCpuMemCpy(void*, const void*, size_t);
void CreateGenericMLValue(const onnxruntime::InputDefList* input_def_list, const AllocatorPtr& alloc,
                          const std::string& name_input, py::object& value, OrtValue* p_mlvalue,
                          bool accept_only_numpy_array = false, bool use_numpy_data_memory = true, MemCpyFunc mem_cpy_to_device = CpuToCpuMemCpy);

void GetPyObjFromTensor(const Tensor& rtensor, py::object& obj,
                        const DataTransferManager* data_transfer_manager = nullptr,
                        const std::unordered_map<OrtDevice::DeviceType, MemCpyFunc>* mem_cpy_to_host_functions = nullptr);

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
