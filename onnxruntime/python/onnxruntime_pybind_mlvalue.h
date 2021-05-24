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

extern const char* PYTHON_ORTVALUE_OBJECT_NAME;
extern const char* PYTHON_ORTVALUE_NATIVE_OBJECT_ATTR;

bool IsNumericNumpyType(int npy_type);

bool IsNumericNumpyArray(py::object& py_object);

int OnnxRuntimeTensorToNumpyType(const DataTypeImpl* tensor_type);

MLDataType NumpyTypeToOnnxRuntimeType(int numpy_type);

using MemCpyFunc = void (*)(void*, const void*, size_t);

void CpuToCpuMemCpy(void*, const void*, size_t);

void AddTensorAsPyObj(const OrtValue& val, std::vector<pybind11::object>& pyobjs,
                      const DataTransferManager* data_transfer_manager,
                      const std::unordered_map<OrtDevice::DeviceType, MemCpyFunc>* mem_cpy_to_host_functions);

void AddNonTensorAsPyObj(const OrtValue& val, std::vector<pybind11::object>& pyobjs,
                         const DataTransferManager* data_transfer_manager,
                         const std::unordered_map<OrtDevice::DeviceType, MemCpyFunc>* mem_cpy_to_host_functions);


#ifdef USE_CUDA

void CpuToCudaMemCpy(void* dst, const void* src, size_t num_bytes);

void CudaToCpuMemCpy(void* dst, const void* src, size_t num_bytes);

const std::unordered_map<OrtDevice::DeviceType, MemCpyFunc>* GetCudaToHostMemCpyFunction();

bool IsCudaDeviceIdValid(const onnxruntime::logging::Logger& logger, int id);

AllocatorPtr GetCudaAllocator(OrtDevice::DeviceId id);

#endif

#ifdef USE_OPENVINO
ProviderInfo_OpenVINO* GetProviderInfo_OpenVINO();
#endif

#ifdef USE_ROCM

bool IsRocmDeviceIdValid(const onnxruntime::logging::Logger& logger, int id);

AllocatorPtr GetRocmAllocator(OrtDevice::DeviceId id);

void CpuToRocmMemCpy(void* dst, const void* src, size_t num_bytes);

void RocmToCpuMemCpy(void* dst, const void* src, size_t num_bytes);

const std::unordered_map<OrtDevice::DeviceType, MemCpyFunc>* GetRocmToHostMemCpyFunction();

#endif


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
