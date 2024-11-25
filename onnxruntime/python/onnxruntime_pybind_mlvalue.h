// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "onnxruntime_pybind.h"  // must use this for the include of <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/common/logging/sinks/cerr_sink.h"
#include "core/session/environment.h"
#include "core/framework/ort_value.h"
#include "core/session/inference_session.h"

#include <variant>

PYBIND11_MAKE_OPAQUE(std::vector<OrtValue>);

namespace onnxruntime {
namespace python {

extern const char* PYTHON_ORTVALUE_OBJECT_NAME;
extern const char* PYTHON_ORTVALUE_NATIVE_OBJECT_ATTR;

bool IsNumericNumpyArray(const pybind11::object& py_object);

bool IsNumpyArray(pybind11::object& obj);

int GetNumpyArrayType(const pybind11::object& obj);

bool IsNumericDType(const pybind11::dtype& dtype);

TensorShape GetShape(const pybind11::array& arr);

int OnnxRuntimeTensorToNumpyType(const DataTypeImpl* tensor_type);

MLDataType NumpyTypeToOnnxRuntimeTensorType(int numpy_type);

MLDataType OnnxTypeToOnnxRuntimeTensorType(int onnx_element_type);

using MemCpyFunc = void (*)(void*, const void*, size_t);

using DataTransferAlternative = std::variant<const DataTransferManager*, MemCpyFunc>;

void CpuToCpuMemCpy(void*, const void*, size_t);

void CopyDataToTensor(const pybind11::array& py_array, int npy_type, Tensor& tensor, MemCpyFunc mem_cpy_to_device = CpuToCpuMemCpy);

pybind11::object AddTensorAsPyObj(const OrtValue& val, const DataTransferManager* data_transfer_manager,
                                  const std::unordered_map<OrtDevice::DeviceType, MemCpyFunc>* mem_cpy_to_host_functions);

pybind11::object GetPyObjectFromSparseTensor(size_t pos, const OrtValue& ort_value, const DataTransferManager* data_transfer_manager);

pybind11::object AddNonTensorAsPyObj(const OrtValue& val,
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
                          const std::string& name_input, const pybind11::object& value, OrtValue* p_mlvalue,
                          bool accept_only_numpy_array = false, bool use_numpy_data_memory = true, MemCpyFunc mem_cpy_to_device = CpuToCpuMemCpy);

pybind11::object GetPyObjFromTensor(const OrtValue& rtensor,
                                    const DataTransferManager* data_transfer_manager = nullptr,
                                    const std::unordered_map<OrtDevice::DeviceType, MemCpyFunc>* mem_cpy_to_host_functions = nullptr);

// The below two functions are used to convert OrtValue to numpy arrays

/// <summary>
/// This function operates on string tensors. Strings are always
/// copied to python and converted to UTF-16/UCS-4/32 depending on the platform.
/// This is accomplished using py::cast()
///
/// It is an error to pass a non-tensor or a non-string tensor to this function.
/// </summary>
/// <param name="tensor">Tensor that contains strings</param>
/// <returns>py::array object</returns>
pybind11::array StringTensorToNumpyArray(const Tensor& tensor);

/// <summary>
/// Creates a numpy array with shape over OrtValue memory. Numpy array
/// does not own the memory, but it holds a copy or OrtValue in a py::capsule.
/// OrtValue is destroyed when the numpy array is garbage collected.
/// This is used when the OrtValue memory is on CPU.
/// </summary>
/// <param name="ort_value">OrtValue with data</param>
/// <returns>numpy array</returns>
pybind11::array PrimitiveTensorToNumpyOverOrtValue(const OrtValue& ort_value);

/// <summary>
/// Creates a numpy array with shape with a copy of OrtValue data.
/// This function is used when the OrtValue memory is not on CPU.
/// </summary>
/// <param name="ort_value">Source memory that is not on CPU.</param>
/// <param name="data_transfer">a variant encapsulating alternatives for copying data</param>
/// <returns></returns>
pybind11::array PrimitiveTensorToNumpyFromDevice(const OrtValue& ort_value,
                                                 const DataTransferAlternative& data_transfer);

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
