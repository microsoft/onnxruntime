// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "onnxruntime_pybind.h"

#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/common/logging/sinks/cerr_sink.h"
#include "core/session/environment.h"
#include "core/framework/ort_value.h"
#include "core/session/inference_session.h"

#include <variant>

// Use the nanobind namespace
namespace nb = nanobind;

// 1. Use NB_MAKE_OPAQUE for opaque type registration
NB_MAKE_OPAQUE(std::vector<OrtValue>);

namespace onnxruntime {
namespace python {

extern const char* PYTHON_ORTVALUE_OBJECT_NAME;
extern const char* PYTHON_ORTVALUE_NATIVE_OBJECT_ATTR;

// 2. Update all function signatures to use nanobind types (nb::object, nb::dtype, nb::ndarray)
bool IsNumericNumpyArray(const nb::object& py_object);

bool IsNumpyArray(nb::object& obj);

int GetNumpyArrayType(const nb::object& obj);

//bool IsNumericDType(const nb::dtype& dtype);

TensorShape GetShape(const nb::ndarray<>& arr);

int OnnxRuntimeTensorToNumpyType(const DataTypeImpl* tensor_type);

MLDataType NumpyTypeToOnnxRuntimeTensorType(int numpy_type);

MLDataType OnnxTypeToOnnxRuntimeTensorType(int onnx_element_type);

using MemCpyFunc = void (*)(void*, const void*, size_t);

using DataTransferAlternative = std::variant<const DataTransferManager*, MemCpyFunc>;

void CpuToCpuMemCpy(void*, const void*, size_t);

void CopyDataToTensor(const nb::ndarray<>& py_array, int npy_type, Tensor& tensor, MemCpyFunc mem_cpy_to_device = CpuToCpuMemCpy);

nb::object AddTensorAsPyObj(const OrtValue& val, const DataTransferManager* data_transfer_manager,
                            const std::unordered_map<OrtDevice::DeviceType, MemCpyFunc>* mem_cpy_to_host_functions);

nb::object GetPyObjectFromSparseTensor(size_t pos, const OrtValue& ort_value, const DataTransferManager* data_transfer_manager);

nb::object AddNonTensorAsPyObj(const OrtValue& val,
                               const DataTransferManager* data_transfer_manager,
                               const std::unordered_map<OrtDevice::DeviceType, MemCpyFunc>* mem_cpy_to_host_functions);

OrtMemoryInfo GetMemoryInfoPerDeviceType(const OrtDevice& ort_device);

int32_t GetTensorProtoType(const OrtValue& ort_value);

#ifdef USE_CUDA

void CpuToCudaMemCpy(void* dst, const void* src, size_t num_bytes);

void CudaToCpuMemCpy(void* dst, const void* src, size_t num_bytes);

const std::unordered_map<OrtDevice::DeviceType, MemCpyFunc>* GetCudaToHostMemCpyFunction();

bool IsCudaDeviceIdValid(const onnxruntime::logging::Logger& logger, int id);

AllocatorPtr GetCudaAllocator(OrtDevice::DeviceId id);

std::unique_ptr<IDataTransfer> GetGPUDataTransfer();

#endif

#ifdef USE_DML

AllocatorPtr GetDmlAllocator(OrtDevice::DeviceId id);

void CpuToDmlMemCpy(void* dst, const void* src, size_t num_bytes);

void DmlToCpuMemCpy(void* dst, const void* src, size_t num_bytes);

const std::unordered_map<OrtDevice::DeviceType, MemCpyFunc>* GetDmlToHostMemCpyFunction();

#endif

#ifdef USE_MIGRAPHX

void CpuToMIGraphXMemCpy(void* dst, const void* src, size_t num_bytes);

void MIGraphXToCpuMemCpy(void* dst, const void* src, size_t num_bytes);

const std::unordered_map<OrtDevice::DeviceType, MemCpyFunc>* GetMIGraphXToHostMemCpyFunction();

AllocatorPtr GetMIGraphXAllocator(OrtDevice::DeviceId id);

#endif

#ifdef USE_CANN

void CpuToCannMemCpy(void* dst, const void* src, size_t num_bytes);

void CannToCpuMemCpy(void* dst, const void* src, size_t num_bytes);

const std::unordered_map<OrtDevice::DeviceType, MemCpyFunc>* GetCannToHostMemCpyFunction();

bool IsCannDeviceIdValid(const onnxruntime::logging::Logger& logger, int id);

AllocatorPtr GetCannAllocator(OrtDevice::DeviceId id);

#endif

#ifdef USE_ROCM

bool IsRocmDeviceIdValid(const onnxruntime::logging::Logger& logger, int id);

AllocatorPtr GetRocmAllocator(OrtDevice::DeviceId id);

void CpuToRocmMemCpy(void* dst, const void* src, size_t num_bytes);

void RocmToCpuMemCpy(void* dst, const void* src, size_t num_bytes);

const std::unordered_map<OrtDevice::DeviceType, MemCpyFunc>* GetRocmToHostMemCpyFunction();

#endif

void CreateGenericMLValue(const onnxruntime::InputDefList* input_def_list, const AllocatorPtr& alloc,
                          const std::string& name_input, const nb::object& value, OrtValue* p_mlvalue,
                          bool accept_only_numpy_array = false, bool use_numpy_data_memory = true, MemCpyFunc mem_cpy_to_device = CpuToCpuMemCpy);

nb::object GetPyObjFromTensor(const OrtValue& rtensor,
                              const DataTransferManager* data_transfer_manager = nullptr,
                              const std::unordered_map<OrtDevice::DeviceType, MemCpyFunc>* mem_cpy_to_host_functions = nullptr);


nb::ndarray<void> StringTensorToNumpyArray(const Tensor& tensor);

nb::ndarray<void> PrimitiveTensorToNumpyOverOrtValue(const OrtValue& ort_value);

nb::ndarray<> PrimitiveTensorToNumpyFromDevice(const OrtValue& ort_value,
                                             const DataTransferAlternative& data_transfer);

// This custom deleter for CPython objects is library-agnostic and does not need to change.
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