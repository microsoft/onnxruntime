// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "python/onnxruntime_pybind_exceptions.h"
#include "python/onnxruntime_pybind_mlvalue.h"
#include "python/onnxruntime_pybind_state_common.h"

#define NO_IMPORT_ARRAY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL onnxruntime_python_ARRAY_API
#include "python/numpy_helper.h"

#include "core/framework/ort_value.h"
#include "core/framework/tensor.h"
#include "core/framework/sparse_tensor.h"
#include "core/framework/TensorSeq.h"
#include "core/common/inlined_containers.h"
namespace onnxruntime {
namespace python {

namespace py = pybind11;

namespace {
bool IsWebGpuBuffer(const OrtValue& ort_value) {
  return ort_value.IsTensor() &&
         ort_value.Get<Tensor>().Location().name == WEBGPU_BUFFER;
}

void RetainNumpyOwner(OrtValue& ort_value, const py::object& array) {
  OrtValue tensor_owner = ort_value;
  auto* tensor = tensor_owner.GetMutable<Tensor>();
  auto type = tensor_owner.Type();
  // The original tensor may itself own a copied NumPy array. Release both
  // owners under the GIL, without blocking an ORT worker that another thread is joining.
  auto owners = py::make_tuple(py::cast(tensor_owner, py::return_value_policy::copy),
                               tensor->OwnsBuffer() ? py::none() : array);
  auto owner = std::shared_ptr<PyObject>(owners.release().ptr(), ReleasePythonObject);
  ort_value.Init(tensor, type, [owner = std::move(owner)](void*) mutable { owner.reset(); });
}

std::unique_ptr<OrtValue> OrtValueFromShapeAndType(const std::vector<int64_t>& shape,
                                                   MLDataType element_type,
                                                   const OrtDevice& device) {
  AllocatorPtr allocator;

  if (strcmp(GetDeviceName(device), CPU) == 0) {
    allocator = GetAllocator();
  } else {
#if !defined(ORT_MINIMAL_BUILD)
    // prefer a shared allocator from the environment.
    // these are provided by plugin EPs or custom allocators explicitly registered by the user.
    allocator = GetSharedAllocator(device);
#endif

    if (!allocator) {
      if (strcmp(GetDeviceName(device), CUDA) == 0) {
#ifdef USE_CUDA
        if (!IsCudaDeviceIdValid(logging::LoggingManager::DefaultLogger(), device.Id())) {
          throw std::runtime_error("The provided device id doesn't match any available GPUs on the machine.");
        }

        allocator = GetCudaAllocator(device.Id());
#else
        throw std::runtime_error(
            "Can't allocate memory on the CUDA device using this package of OnnxRuntime. "
            "Please use the CUDA package of OnnxRuntime to use this feature.");
#endif
      } else if (strcmp(GetDeviceName(device), HIP) == 0) {
#if USE_MIGRAPHX
        allocator = GetMIGraphXAllocator(device.Id());
#else
        throw std::runtime_error(
            "Can't allocate memory on the AMD device using this package of OnnxRuntime. "
            "Please use the ROCm package of OnnxRuntime to use this feature.");
#endif
      } else if (strcmp(GetDeviceName(device), DML) == 0) {
#if USE_DML
        allocator = GetDmlAllocator(device.Id());
#else
        throw std::runtime_error(
            "Can't allocate memory on the DirectML device using this package of OnnxRuntime. "
            "Please use the DirectML package of OnnxRuntime to use this feature.");
#endif
      }
    }

    if (!allocator) {
      throw std::runtime_error("Unsupported device: Cannot place the OrtValue on this device");
    }
  }

  auto ml_value = std::make_unique<OrtValue>();
  Tensor::InitOrtValue(element_type, gsl::make_span(shape), std::move(allocator), *ml_value);
  return ml_value;
}

// Allocate an OrtValue using the shared allocator matching the given OrtMemoryInfo.
// This allows callers to specify the exact memory type (e.g. HOST_ACCESSIBLE) rather than
// relying on OrtDevice.make() which always uses DEFAULT.
//
// Uses the full OrtMemoryInfo for the lookup (including mem_type) rather than just the OrtDevice,
// because the registered allocator's OrtMemoryInfo has a specific mem_type (e.g. OrtMemTypeCPU
// for HOST_ACCESSIBLE) that must match for FindExistingAllocator to succeed.
std::unique_ptr<OrtValue> OrtValueFromShapeAndTypeWithMemoryInfo(const std::vector<int64_t>& shape,
                                                                 MLDataType element_type,
                                                                 const OrtMemoryInfo& memory_info) {
  auto& env = GetOrtEnv()->GetEnvironment();
  AllocatorPtr allocator = env.GetRegisteredSharedAllocator(memory_info);

  if (!allocator) {
    throw std::runtime_error("No shared allocator found for: " + memory_info.ToString());
  }

  auto ml_value = std::make_unique<OrtValue>();
  Tensor::InitOrtValue(element_type, gsl::make_span(shape), std::move(allocator), *ml_value);
  return ml_value;
}

}  // namespace

void addOrtValueMethods(pybind11::module& m) {
  py::class_<OrtValue> ortvalue_binding(m, "OrtValue");
  ortvalue_binding
      // Share contiguous CPU NumPy storage; copy other inputs to the requested device.
      .def_static("ortvalue_from_numpy", [](const py::object& array_on_cpu, const OrtDevice& device) {
        if (!IsNumericNumpyArray(array_on_cpu)) {
          throw std::runtime_error("Creation of OrtValues is currently only supported from non-string numpy arrays");
        }

        auto ml_value = std::make_unique<OrtValue>();

        // The tensor's memory is allocated on the CPU
        if (device.Type() == OrtDevice::CPU) {
          // InputDeflist is null because OrtValue creation is not tied to a specific model
          // Likewise, there is no need to specify the name (as the name was previously used to lookup the def list)

          CreateGenericMLValue(nullptr, GetAllocator(), "", array_on_cpu, ml_value.get(), true);
          RetainNumpyOwner(*ml_value, array_on_cpu);
        } else if (device.Type() == OrtDevice::GPU) {
#if USE_DML
          if (device.Vendor() == OrtDevice::VendorIds::MICROSOFT) {
            // InputDeflist is null because OrtValue creation is not tied to a specific model
            // Likewise, there is no need to specify the name (as the name was previously used to lookup the def list)
            // TODO: Add check to ensure that string arrays are not passed - we currently don't support string tensors
            // in DML
            CreateGenericMLValue(
                nullptr, GetDmlAllocator(device.Id()), "", array_on_cpu, ml_value.get(), true, false, CpuToDmlMemCpy);
          } else
#endif
#ifdef USE_CUDA
              if (device.Vendor() == OrtDevice::VendorIds::NVIDIA) {
            if (!IsCudaDeviceIdValid(logging::LoggingManager::DefaultLogger(), device.Id())) {
              throw std::runtime_error("The provided device id doesn't match any available GPUs on the machine.");
            }

            // InputDeflist is null because OrtValue creation is not tied to a specific model
            // Likewise, there is no need to specify the name (as the name was previously used to lookup the def list)
            // TODO: Add check to ensure that string arrays are not passed - we currently don't support string tensors
            // in CUDA
            CreateGenericMLValue(nullptr, GetCudaAllocator(device.Id()), "", array_on_cpu, ml_value.get(),
                                 true, false, CpuToCudaMemCpy);
          } else
#endif
#if USE_MIGRAPHX
              if (device.Vendor() == OrtDevice::VendorIds::AMD) {
            // InputDeflist is null because OrtValue creation is not tied to a specific model
            // Likewise, there is no need to specify the name (as the name was previously used to lookup the def list)
            // TODO: Add check to ensure that string arrays are not passed - we currently don't support string tensors
            // in MIGraphX
            CreateGenericMLValue(nullptr, GetMIGraphXAllocator(device.Id()), "", array_on_cpu, ml_value.get(),
                                 true, false, CpuToMIGraphXMemCpy);
          } else
#endif
          {
            // see if we can do the copy with an allocator and IDataTransfer registered by a plugin EP
            auto allocator = GetSharedAllocator(device);
            auto cpu_to_device_copy_fn = allocator ? CreateDataTransferMemCpy(OrtDevice{}, device) : nullptr;
            if (cpu_to_device_copy_fn) {
              CreateGenericMLValue(nullptr, allocator, "", array_on_cpu, ml_value.get(), true, false,
                                   cpu_to_device_copy_fn);
            } else {
              throw std::runtime_error(
                  "Can't allocate memory on the device using this package of OnnxRuntime. "
                  "Please use the appropriate package of OnnxRuntime for your hardware to use this feature.");
            }
          }
        } else if (device.Type() == OrtDevice::NPU && device.Vendor() == OrtDevice::VendorIds::HUAWEI) {
#ifdef USE_CANN
          if (!IsCannDeviceIdValid(logging::LoggingManager::DefaultLogger(), device.Id())) {
            throw std::runtime_error("The provided device id doesn't match any available NPUs on the machine.");
          }

          CreateGenericMLValue(nullptr, GetCannAllocator(device.Id()), "", array_on_cpu, ml_value.get(),
                               true, false, CpuToCannMemCpy);
#else
          throw std::runtime_error(
              "Can't allocate memory on the CANN device using this package of OnnxRuntime. "
              "Please use the CANN package of OnnxRuntime to use this feature.");
#endif
        } else {
          throw std::runtime_error("Unsupported device: Cannot place the OrtValue on this device");
        }

        return ml_value;
      })
      .def("update_inplace", [](OrtValue* ml_value, const py::array& py_values) {
        if (!IsNumericNumpyArray(py_values)) {
          throw std::runtime_error("Inplace update of OrtValues is currently only supported from non-string numpy arrays");
        }

        if (py_values.size() != ml_value->Get<Tensor>().Shape().Size()) {
          throw std::runtime_error("The input size of numpy arrays does not match the size of the OrtValue.");
        }

        auto values_type = GetNumpyArrayType(py_values);
        const auto device = ml_value->Get<Tensor>().Location().device;
        if (device.Type() == OrtDevice::CPU) {
          onnxruntime::python::CopyDataToTensor(
              py_values,
              values_type,
              *(ml_value->GetMutable<Tensor>()),
              CpuToCpuMemCpy);
        } else if (device.Type() == OrtDevice::GPU) {
#ifdef USE_CUDA
          if (device.Vendor() == OrtDevice::VendorIds::NVIDIA) {
            MemCpyFunc cpu_to_device_copy_fn = CpuToCudaMemCpy;
            if (TryGetProviderInfo_CUDA() != nullptr) {
              if (!IsCudaDeviceIdValid(logging::LoggingManager::DefaultLogger(), device.Id())) {
                throw std::runtime_error("The provided device id doesn't match any available GPUs on the machine.");
              }
            } else {
              cpu_to_device_copy_fn = CreateDataTransferMemCpy(OrtDevice{}, device);
              if (!cpu_to_device_copy_fn) {
                throw std::runtime_error(
                    "Unsupported GPU device: Cannot find the supported GPU device.");
              }
            }

            onnxruntime::python::CopyDataToTensor(
                py_values,
                values_type,
                *(ml_value->GetMutable<Tensor>()),
                cpu_to_device_copy_fn);
          } else
#endif
#if USE_MIGRAPHX
              if (device.Vendor() == OrtDevice::VendorIds::AMD) {
            onnxruntime::python::CopyDataToTensor(
                py_values,
                values_type,
                *(ml_value->GetMutable<Tensor>()),
                CpuToMIGraphXMemCpy);
          } else
#endif
#if USE_DML
              if (device.Vendor() == OrtDevice::VendorIds::MICROSOFT) {
            onnxruntime::python::CopyDataToTensor(
                py_values,
                values_type,
                *(ml_value->GetMutable<Tensor>()),
                CpuToDmlMemCpy);
          } else
#endif
          {
            // see if we can do the copy with an allocator and IDataTransfer registered by a plugin EP
            auto allocator = GetSharedAllocator(device);
            auto cpu_to_device_copy_fn = allocator ? CreateDataTransferMemCpy(OrtDevice{}, device) : nullptr;
            if (cpu_to_device_copy_fn) {
              onnxruntime::python::CopyDataToTensor(py_values, values_type, *(ml_value->GetMutable<Tensor>()),
                                                    cpu_to_device_copy_fn);
            } else {
              throw std::runtime_error(
                  "Unsupported GPU device: Cannot find the supported GPU device.");
            }
          }
        } else if (device.Type() == OrtDevice::DML) {
#if USE_DML
          onnxruntime::python::CopyDataToTensor(
              py_values,
              values_type,
              *(ml_value->GetMutable<Tensor>()),
              CpuToDmlMemCpy);
#else
          throw std::runtime_error(
              "Unsupported GPU device: Cannot find the supported GPU device.");
#endif
        } else {
          throw std::runtime_error("Unsupported device: Cannot update the OrtValue on this device");
        }
      })
      .def("update_inplace", [](OrtValue* ml_value, const OrtValue& source) {
        python::UpdateOrtValueInplace(*ml_value, source);
      })
      // Create an ortvalue value on top of the numpy array, but interpret the data
      // as a different type with the same element size.
      .def_static("ortvalue_from_numpy_with_onnx_type", [](py::array& data, int32_t onnx_element_type) -> std::unique_ptr<OrtValue> {
        if (!(data.flags() & py::array::c_style)) {
          throw py::value_error(
              "ortvalue_from_numpy_with_onnx_type requires a C-contiguous NumPy array. "
              "Use numpy.ascontiguousarray(data) before importing.");
        }
        if (!ONNX_NAMESPACE::TensorProto_DataType_IsValid(onnx_element_type)) {
          ORT_THROW("Not a valid ONNX Tensor data type: ", onnx_element_type);
        }

        const auto element_type = OnnxTypeToOnnxRuntimeTensorType(onnx_element_type);

        const auto element_size = element_type->Size();
        if (narrow<size_t>(data.itemsize()) != element_size) {
          ORT_THROW("Items size in the incoming array: ", data.itemsize(),
                    " specified by onnxtype: ", element_size);
        }

        auto cpu_allocator = GetAllocator();
        auto ort_value = std::make_unique<OrtValue>();
        Tensor::InitOrtValue(element_type, GetShape(data),
                             const_cast<void*>(data.data()), cpu_allocator->Info(), *ort_value);
        RetainNumpyOwner(*ort_value, data);
        return ort_value;
      })
      // Factory method to create an OrtValue from the given shape and numpy element type on the specified device.
      // The memory is left uninitialized
      .def_static("ortvalue_from_shape_and_type", [](const std::vector<int64_t>& shape, py::object& numpy_element_type, const OrtDevice& device) -> std::unique_ptr<OrtValue> {
        PyArray_Descr* dtype;
        if (!PyArray_DescrConverter(numpy_element_type.ptr(), &dtype)) {
          throw std::runtime_error("Not a valid numpy type");
        }

        int type_num = dtype->type_num;
        Py_DECREF(dtype);

        if (!IsNumericNumpyType(type_num)) {
          throw std::runtime_error("Creation of OrtValues is currently only supported from non-string numpy arrays");
        }

        auto element_type = NumpyTypeToOnnxRuntimeTensorType(type_num);
        return OrtValueFromShapeAndType(shape, element_type, device);
      })
      // Factory method to create an OrtValue from the given shape and onnx element type on the specified device.
      // The memory is left uninitialized
      .def_static("ortvalue_from_shape_and_onnx_type", [](const std::vector<int64_t>& shape, int32_t onnx_element_type, const OrtDevice& device) -> std::unique_ptr<OrtValue> {
        if (onnx_element_type == onnx::TensorProto_DataType::TensorProto_DataType_STRING) {
          throw std::runtime_error("Creation of OrtValues is currently only supported from non-string numpy arrays");
        }

        auto element_type = OnnxTypeToOnnxRuntimeTensorType(onnx_element_type);
        return OrtValueFromShapeAndType(shape, element_type, device);
      })
      // Factory methods to create an OrtValue using an OrtMemoryInfo to select the allocator.
      // This enables allocation with a specific memory type (e.g. HOST_ACCESSIBLE) from plugin EPs.
      .def_static("ortvalue_from_shape_and_type_for_memory_info", [](const std::vector<int64_t>& shape, py::object& numpy_element_type, const OrtMemoryInfo& memory_info) -> std::unique_ptr<OrtValue> {
        PyArray_Descr* dtype;
        if (!PyArray_DescrConverter(numpy_element_type.ptr(), &dtype)) {
          throw std::runtime_error("Not a valid numpy type");
        }

        int type_num = dtype->type_num;
        Py_DECREF(dtype);

        if (!IsNumericNumpyType(type_num)) {
          throw std::runtime_error("Creation of OrtValues is currently only supported from non-string numpy arrays");
        }

        auto element_type = NumpyTypeToOnnxRuntimeTensorType(type_num);
        return OrtValueFromShapeAndTypeWithMemoryInfo(shape, element_type, memory_info);
      })
      .def_static("ortvalue_from_shape_and_onnx_type_for_memory_info", [](const std::vector<int64_t>& shape, int32_t onnx_element_type, const OrtMemoryInfo& memory_info) -> std::unique_ptr<OrtValue> {
        if (onnx_element_type == onnx::TensorProto_DataType::TensorProto_DataType_STRING) {
          throw std::runtime_error("Creation of OrtValues is currently only supported from non-string numpy arrays");
        }

        auto element_type = OnnxTypeToOnnxRuntimeTensorType(onnx_element_type);
        return OrtValueFromShapeAndTypeWithMemoryInfo(shape, element_type, memory_info);
      })

#if !defined(DISABLE_SPARSE_TENSORS)
      .def_static("ort_value_from_sparse_tensor", [](const PySparseTensor* py_sparse_tensor) -> std::unique_ptr<OrtValue> {
        return py_sparse_tensor->AsOrtValue();
      })
      // This will create a copy of OrtValue(cheap) and will return as a separate SparseTensor object
      .def("as_sparse_tensor", [](const OrtValue* ort_value) -> std::unique_ptr<PySparseTensor> {
        if (!ort_value->IsSparseTensor()) {
          ORT_THROW("This OrtValue does not contain SparseTensor. Check data_type() value.");
        }
        return std::make_unique<PySparseTensor>(*ort_value);
      })
#endif
      // Get a pointer to Tensor data
      .def("data_ptr", [](OrtValue* ml_value) -> uintptr_t {
        // TODO: Assumes that the OrtValue is a Tensor, make this generic to handle non-Tensors
        ORT_ENFORCE(ml_value->IsTensor(), "Only OrtValues that are Tensors are currently supported");

        auto* tensor = ml_value->GetMutable<Tensor>();

        if (tensor->Shape().Size() == 0) {
          return 0;
        }

        // Should cover x86 and x64 platforms
        return reinterpret_cast<uintptr_t>(tensor->MutableDataRaw());
      })
      .def("device_name", [](const OrtValue* ort_value) -> std::string {
        if (ort_value->IsTensor()) {
          if (IsWebGpuBuffer(*ort_value)) {
            return "webgpu";
          }
          return std::string(GetDeviceName(ort_value->Get<Tensor>().Location().device));
        }
#if !defined(DISABLE_SPARSE_TENSORS)
        else if (ort_value->IsSparseTensor()) {
          return std::string(GetDeviceName(ort_value->Get<SparseTensor>().Location().device));
        }

        ORT_THROW("Only OrtValues that are Tensors/SparseTensors are currently supported");
#else
        ORT_THROW("Only OrtValues that are Tensors are supported in this build");
#endif
      })
      .def("shape", [](const OrtValue* ort_value) -> py::list {
#if !defined(DISABLE_SPARSE_TENSORS)
        // OrtValue can only be a Tensor/SparseTensor, make this generic to handle non-Tensors
        ORT_ENFORCE(ort_value->IsTensor() || ort_value->IsSparseTensor(),
                    "Only OrtValues that are Tensors/SpareTensors are currently supported");

        const auto dims = (ort_value->IsTensor())
                              ? ort_value->Get<Tensor>().Shape().GetDims()
                              : ort_value->Get<SparseTensor>().DenseShape().GetDims();
#else
        ORT_ENFORCE(ort_value->IsTensor(), "Only OrtValues that are Tensors are supported in this build");
        const auto dims = ort_value->Get<Tensor>().Shape().GetDims();
#endif

        py::list shape_arr;
        for (auto dim : dims) {
          // For sequence tensors - we would append a list of dims to the outermost list
          // For now only tensors are supported in OrtValue
          shape_arr.append(dim);
        }

        return shape_arr;
      })
      .def("data_type", [](const OrtValue* ort_value) -> std::string {
        const ONNX_NAMESPACE::TypeProto* type_proto;
        // Handle gutless types first to get the actual type
        if (ort_value->IsTensor()) {
          auto elem_type = ort_value->Get<Tensor>().GetElementType();
          type_proto = DataTypeImpl::TensorTypeFromONNXEnum(elem_type)->GetTypeProto();
#if !defined(DISABLE_SPARSE_TENSORS)
        } else if (ort_value->IsSparseTensor()) {
          auto elem_type = ort_value->Get<SparseTensor>().GetElementType();
          type_proto = DataTypeImpl::SparseTensorTypeFromONNXEnum(elem_type)->GetTypeProto();
#endif
        } else if (ort_value->IsTensorSequence()) {
          auto elem_type = ort_value->Get<TensorSeq>().DataType()->AsPrimitiveDataType()->GetDataType();
          type_proto = DataTypeImpl::SequenceTensorTypeFromONNXEnum(elem_type)->GetTypeProto();
        } else {
          // Plane sequences and maps probably have their specific type
          type_proto = ort_value->Type()->GetTypeProto();
        }

        ORT_ENFORCE(type_proto != nullptr, "Unknown type of OrtValue: ", ort_value->Type());

        return *ONNX_NAMESPACE::Utils::DataTypeUtils::ToType(*type_proto);
      })
      .def("element_type", [](const OrtValue* ort_value) -> int32_t { return GetTensorProtoType(*ort_value); },
           "Returns an integer equal to the ONNX tensor proto type of the tensor or sequence. "
           "This integer is one type defined by ONNX TensorProto_DataType "
           "(such as onnx.TensorProto.FLOAT)."
           "Raises an exception in any other case.")
      .def("tensor_size_in_bytes", [](const OrtValue* ort_value) -> size_t {
        ORT_ENFORCE(ort_value->IsTensor(), "Only OrtValues that are Tensors are currently supported");
        return ort_value->Get<Tensor>().SizeInBytes(); }, "Returns tensor size in bytes.")
      .def("has_value", [](const OrtValue* ort_value) -> bool { return ort_value->IsAllocated(); })
      .def("is_tensor", [](const OrtValue* ort_value) -> bool { return ort_value->IsTensor(); })
      .def("is_sparse_tensor", [](const OrtValue* ort_value) -> bool { return ort_value->IsSparseTensor(); })
      .def("is_tensor_sequence", [](const OrtValue* ort_value) -> bool { return ort_value->IsTensorSequence(); })
      .def("_is_webgpu_buffer", [](const OrtValue* ort_value) -> bool { return IsWebGpuBuffer(*ort_value); })
      // Converts Tensor into a numpy array
      .def("numpy", [](const OrtValue* ml_value) -> py::object {
        ORT_ENFORCE(ml_value->IsTensor(), "Only OrtValues that are Tensors are convertible to Numpy objects");
        [[maybe_unused]] const auto& device = ml_value->Get<Tensor>().Location().device;
#ifdef _MSC_VER
// The switch statement may only contain the 'default' label. In such a case, the MSVC compiler
// will warn about it, and since the warnings are treated as errors, the compilation will break.
// Below pragmas turn off warning generation for this switch only.
#pragma warning(push)
#pragma warning(disable : 4065)
#endif
        switch (device.Vendor()) {
#ifdef USE_CUDA
          case OrtDevice::VendorIds::NVIDIA:
            if (TryGetProviderInfo_CUDA() == nullptr) {
              return GetPyObjFromTensor(*ml_value, nullptr, nullptr,
                                        /*zero_copy_non_owning=*/true);
            }
            return GetPyObjFromTensor(*ml_value, nullptr, GetCudaToHostMemCpyFunction(device),
                                      /*zero_copy_non_owning=*/true);
#endif
#ifdef USE_CANN
          case OrtDevice::VendorIds::HUAWEI:
            return GetPyObjFromTensor(*ml_value, nullptr, GetCannToHostMemCpyFunction(),
                                      /*zero_copy_non_owning=*/true);
#endif

#ifdef USE_DML
          case OrtDevice::VendorIds::MICROSOFT:
            return GetPyObjFromTensor(*ml_value, nullptr, GetDmlToHostMemCpyFunction(device),
                                      /*zero_copy_non_owning=*/true);
#endif
#ifdef USE_MIGRAPHX
          case OrtDevice::VendorIds::AMD:
            return GetPyObjFromTensor(*ml_value, nullptr, GetMIGraphXToHostMemCpyFunction(device),
                                      /*zero_copy_non_owning=*/true);
#endif
          default:
            // OrtValue.numpy() is called by the user who explicitly holds the OrtValue
            // Python object, so the backing memory lifetime is managed externally.
            // zero_copy_non_owning=true is safe here (and required to preserve the
            // zero-copy semantics that OrtValue.numpy() / __array__ rely on).
            return GetPyObjFromTensor(*ml_value, nullptr, nullptr,
                                      /*zero_copy_non_owning=*/true);
        }
#ifdef _MSC_VER
#pragma warning(pop)
#endif
      })
#if defined(ENABLE_DLPACK)
      .def_static("from_dlpack", &FromDlpack, py::arg("data"), py::kw_only(), py::arg("synchronize") = true,
                  "Imports a DLPack object or capsule without copying. Set synchronize=False only when the caller "
                  "arranges producer/consumer stream ordering.")
      .def("__dlpack__", [](OrtValue* ort_value, py::object stream) -> py::object {
        ORT_ENFORCE(!IsWebGpuBuffer(*ort_value), "DLPack export is not supported for WebGPU OrtValues.");
        return py::reinterpret_steal<py::object>(ToDlpack(*ort_value, std::move(stream))); }, py::arg("stream") = py::none(),
           "Returns a DLPack representing the tensor (part of __dlpack__ protocol). "
           "The capsule retains the tensor and its NumPy owner without copying. "
           "GPU exports synchronize the tensor's device unless stream=-1 requests no synchronization.")
      .def("__dlpack_device__", [](const OrtValue* ort_value) -> py::tuple {
            ORT_ENFORCE(ort_value->IsTensor(), "Only tensor type OrtValues are supported");
            ORT_ENFORCE(!IsWebGpuBuffer(*ort_value), "DLPack export is not supported for WebGPU OrtValues.");
            const onnxruntime::Tensor& tensor = ort_value->Get<Tensor>();
            DLDevice device = onnxruntime::dlpack::GetDlpackDevice(*ort_value, tensor.Location().device.Id());
            return py::make_tuple(static_cast<int>(device.device_type), device.device_id); }, "Returns a tuple of integers, (device, device index) (part of __dlpack__ protocol).")
#endif
      ;

  py::class_<std::vector<OrtValue>>(m, "OrtValueVector")
      .def(py::init<>())
      .def("push_back", [](std::vector<OrtValue>* v, const OrtValue& ortvalue) {
        // A standalone OrtValueVector has no parent object, so unlike the vector returned by
        // SessionIOBinding::get_outputs it cannot keep the owning session alive. A WebGPU buffer
        // allocated by a session's EP is freed through GpuBufferAllocator, whose buffer-manager
        // getter captures that EP, so releasing it after the session is gone is a use-after-free.
        // TODO: remove once GpuBufferAllocator owns a reference to whatever owns its BufferManager,
        // at which point a WebGPU OrtValue is safe to hold independently of any session.
        ORT_ENFORCE(!IsWebGpuBuffer(ortvalue),
                    "A WebGPU OrtValue cannot be stored in a standalone OrtValueVector because the "
                    "vector cannot keep the session that allocated the buffer alive. Use IOBinding, "
                    "or IOBinding.get_outputs_as_ortvaluevector() for device-resident outputs.");
        v->push_back(ortvalue);
      })
#if defined(ENABLE_DLPACK)
      .def("push_back", [](std::vector<OrtValue>* v, py::object data, bool synchronize) { v->push_back(FromDlpack(std::move(data), synchronize)); }, "Add an OrtValue imported from a DLPack object or capsule. Set synchronize=False only when arranging stream ordering.", py::arg("dlpack_tensor"), py::kw_only(), py::arg("synchronize") = true)
      .def("push_back_batch", [](std::vector<OrtValue>* v, std::vector<py::object>& torch_tensors, std::vector<int64_t>& data_ptrs, std::vector<py::object>& element_types, const std::vector<std::vector<int64_t>>& shapes, const std::vector<OrtDevice>& devices) {
            for (size_t i = 0; i < torch_tensors.size(); ++i) {
              py::object& element_type = element_types.at(i);
              const std::vector<int64_t>& shape = shapes.at(i);
              int64_t data_ptr = data_ptrs.at(i);

              ORT_ENFORCE(data_ptr, "Pointer to data memory is not valid");

              PyArray_Descr* dtype;
              if (!PyArray_DescrConverter(element_type.ptr(), &dtype)) {
                throw std::runtime_error("Not a valid numpy type");
              }
              int type_num = dtype->type_num;
              Py_DECREF(dtype);

              auto ml_type = NumpyTypeToOnnxRuntimeTensorType(type_num);
              auto device = devices.at(i);
              // This overload wraps foreign (PyTorch) storage by raw address. A WebGPU allocation is
              // an opaque WGPUBuffer handle rather than an addressable pointer, so no torch tensor
              // can back one. The OrtMemoryInfo built below would also be mislabelled, since it uses
              // GetDeviceName rather than the WEBGPU_BUFFER allocator name the binding paths expect.
              ORT_ENFORCE(!(device.Type() == OrtDevice::GPU &&
                            device.Vendor() == OrtDevice::VendorIds::NONE),
                          "OrtValueVector.push_back_batch cannot wrap WebGPU memory: a WebGPU "
                          "allocation is an opaque buffer handle, not a raw data pointer.");
              OrtMemoryInfo info(GetDeviceName(device), OrtDeviceAllocator, device);
              OrtValue ml_value;
              Tensor::InitOrtValue(ml_type, gsl::make_span(shape), reinterpret_cast<void*>(data_ptr), info, ml_value);
              v->push_back(ml_value);
            } }, "Add a batch of OrtValue's by wrapping PyTorch tensors.")
#endif
      .def("reserve", [](std::vector<OrtValue>* v, const size_t len) { v->reserve(len); })
      .def("shrink_to_fit", [](std::vector<OrtValue>* v) { v->shrink_to_fit(); })
      .def("__len__", [](const std::vector<OrtValue>& v) { return v.size(); })
      .def("__iter__", [](const std::vector<OrtValue>& v) { return py::make_iterator(v.cbegin(), v.cend()); }, py::keep_alive<0, 1>())
      .def("__getitem__", [](const std::vector<OrtValue>& v, const size_t idx) { return v.at(idx); }, py::keep_alive<0, 1>())
      .def("element_type_at", [](std::vector<OrtValue>* v, const size_t idx) -> int32_t { return GetTensorProtoType(v->at(idx)); },
           "Returns an integer equal to the ONNX proto type of the tensor at position i. "
           "This integer is one type defined by ONNX TensorProto_DataType "
           "(such as onnx.TensorProto.FLOAT)."
           "Raises an exception in any other case.",
           py::arg("idx"))
#if defined(ENABLE_DLPACK)
      .def("to_dlpacks", [](const std::vector<OrtValue>& v, py::object to_tensor, bool synchronize) -> py::list {
            if (!to_tensor.is_none() && !PyCallable_Check(to_tensor.ptr())) {
              throw py::type_error("OrtValueVector.to_dlpacks expects a callable consumer or None.");
            }
            // Callbacks can mutate the original vector, including reallocating its storage.
            auto values = v;
            InlinedHashMap<int, InlinedHashSet<int>> synchronized_devices;
            py::list capsules;
            for (auto& value : values) {
              ORT_ENFORCE(value.IsTensor(), "DLPack export requires a dense tensor OrtValue; initialize the value before exporting.");
              ORT_ENFORCE(!IsWebGpuBuffer(value), "DLPack export is not supported for WebGPU OrtValues.");
              const auto device = dlpack::GetDlpackDevice(value, value.Get<Tensor>().Location().device.Id());
              py::object stream = py::none();
              if (device.device_type == kDLCUDA || device.device_type == kDLROCM) {
                if (!synchronize || !synchronized_devices[device.device_type].insert(device.device_id).second) {
                  stream = py::int_(-1);
                }
              }
              capsules.append(py::reinterpret_steal<py::object>(ToDlpack(value, std::move(stream))));
            }
            py::list results;
            for (auto capsule : capsules) {
              results.append(to_tensor.is_none() ? capsule : to_tensor(capsule));
            }
            return results; },
           R"pbdoc(Exports a snapshot of the vector as independently owned DLPack capsules.
If `to_tensor` is callable, returns the result of calling it with each capsule.
Consumer exceptions propagate unchanged. Unconsumed capsules release their tensor
references when destroyed, including when conversion fails.

:param to_tensor: this function takes a capsule holding a pointer onto a DLPack structure and returns
    a new tensor which becomes the new owner of the data. This function takes one python object and
    returns a new python object. It fits the same signature as `torch.from_dlpack`,
    if None, the method returns a capsule for every new DLPack structure.
:param synchronize: synchronize each distinct GPU device once before preparing its
    capsules. Set False only when the caller arranges producer/consumer stream ordering.
:return: a list containing the new tensors or the new capsules if *to_tensor* is None

Use `ort_values.to_dlpacks(torch.from_dlpack)` to avoid creating a Python OrtValue
wrapper for every tensor. Each capsule may be consumed only once. All capsules
are prepared before any consumer callback runs.
)pbdoc",
           py::arg("to_tensor"), py::kw_only(), py::arg("synchronize") = true)
#endif
      ;
}

}  // namespace python
}  // namespace onnxruntime
