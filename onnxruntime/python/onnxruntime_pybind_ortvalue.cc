// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "python/onnxruntime_pybind_exceptions.h"
#include "python/onnxruntime_pybind_mlvalue.h"
#include "python/onnxruntime_pybind_state_common.h"

#define NO_IMPORT_ARRAY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL onnxruntime_python_ARRAY_API
#include <numpy/arrayobject.h>
#include "python/numpy_helper.h"

#include "core/framework/ort_value.h"
#include "core/framework/tensor.h"
#include "core/framework/sparse_tensor.h"
#include "core/framework/TensorSeq.h"

namespace onnxruntime {
namespace python {

namespace py = pybind11;

void addOrtValueMethods(pybind11::module& m) {
  py::class_<OrtValue> ortvalue_binding(m, "OrtValue");
  ortvalue_binding
      // Factory method to create an OrtValue (Tensor) from the given Numpy object
      // The Tensor allocates and manages its own memory (on the specified device) and copies data from the Numpy data buffer
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
        } else if (device.Type() == OrtDevice::GPU) {
      // The tensor's memory is allocated on CUDA

#ifdef USE_CUDA
          if (!IsCudaDeviceIdValid(logging::LoggingManager::DefaultLogger(), device.Id())) {
            throw std::runtime_error("The provided device id doesn't match any available GPUs on the machine.");
          }

          // InputDeflist is null because OrtValue creation is not tied to a specific model
          // Likewise, there is no need to specify the name (as the name was previously used to lookup the def list)
          // TODO: Add check to ensure that string arrays are not passed - we currently don't support string tensors in CUDA
          CreateGenericMLValue(nullptr, GetCudaAllocator(device.Id()), "", array_on_cpu, ml_value.get(), true, false, CpuToCudaMemCpy);
#elif USE_ROCM
      if (!IsRocmDeviceIdValid(logging::LoggingManager::DefaultLogger(), device.Id())) {
        throw std::runtime_error("The provided device id doesn't match any available GPUs on the machine.");
      }

      // InputDeflist is null because OrtValue creation is not tied to a specific model
      // Likewise, there is no need to specify the name (as the name was previously used to lookup the def list)
      // TODO: Add check to ensure that string arrays are not passed - we currently don't support string tensors in CUDA
      CreateGenericMLValue(nullptr, GetRocmAllocator(device.Id()), "", array_on_cpu, ml_value.get(), true, false, CpuToRocmMemCpy);

#else
      throw std::runtime_error(
          "Can't allocate memory on the CUDA device using this package of OnnxRuntime. "
          "Please use the CUDA package of OnnxRuntime to use this feature.");
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
          if (!IsCudaDeviceIdValid(logging::LoggingManager::DefaultLogger(), device.Id())) {
            throw std::runtime_error("The provided device id doesn't match any available GPUs on the machine.");
          }

          onnxruntime::python::CopyDataToTensor(
            py_values,
            values_type,
            *(ml_value->GetMutable<Tensor>()),
            CpuToCudaMemCpy);
#elif USE_ROCM
          if (!IsRocmDeviceIdValid(logging::LoggingManager::DefaultLogger(), device.Id())) {
            throw std::runtime_error("The provided device id doesn't match any available GPUs on the machine.");
          }

          onnxruntime::python::CopyDataToTensor(
            py_values,
            values_type,
            *(ml_value->GetMutable<Tensor>()),
            CpuToRocmMemCpy);
#else
        throw std::runtime_error(
            "Unsupported GPU device: Cannot find the supported GPU device.");
#endif
        } else {
          throw std::runtime_error("Unsupported device: Cannot update the OrtValue on this device");
        }
      })
      // Factory method to create an OrtValue (Tensor) from the given shape and element type with memory on the specified device
      // The memory is left uninitialized
      .def_static("ortvalue_from_shape_and_type", [](const std::vector<int64_t>& shape, py::object& element_type, const OrtDevice& device) {
        PyArray_Descr* dtype;
        if (!PyArray_DescrConverter(element_type.ptr(), &dtype)) {
          throw std::runtime_error("Not a valid numpy type");
        }

        int type_num = dtype->type_num;
        Py_DECREF(dtype);

        if (!IsNumericNumpyType(type_num)) {
          throw std::runtime_error("Creation of OrtValues is currently only supported from non-string numpy arrays");
        }

        AllocatorPtr allocator;
        if (strcmp(GetDeviceName(device), CPU) == 0) {
          allocator = GetAllocator();
        } else if (strcmp(GetDeviceName(device), CUDA) == 0) {
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
        } else {
          throw std::runtime_error("Unsupported device: Cannot place the OrtValue on this device");
        }

        auto ml_value = std::make_unique<OrtValue>();
        auto ml_type = NumpyTypeToOnnxRuntimeType(type_num);
        Tensor::InitOrtValue(ml_type, gsl::make_span(shape), std::move(allocator), *ml_value);
        return ml_value;
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
      .def("data_ptr", [](OrtValue* ml_value) -> int64_t {
        // TODO: Assumes that the OrtValue is a Tensor, make this generic to handle non-Tensors
        ORT_ENFORCE(ml_value->IsTensor(), "Only OrtValues that are Tensors are currently supported");

        auto* tensor = ml_value->GetMutable<Tensor>();

        if (tensor->Shape().Size() == 0) {
          return 0;
        }

        // Should cover x86 and x64 platforms
        return reinterpret_cast<int64_t>(tensor->MutableDataRaw());
      })
      .def("device_name", [](const OrtValue* ort_value) -> std::string {
        if (ort_value->IsTensor()) {
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
        py::list shape_arr;
#if !defined(DISABLE_SPARSE_TENSORS)
        // OrtValue can only be a Tensor/SparseTensor, make this generic to handle non-Tensors
        ORT_ENFORCE(ort_value->IsTensor() || ort_value->IsSparseTensor(),
                    "Only OrtValues that are Tensors/SpareTensors are currently supported");

        const auto& dims = (ort_value->IsTensor())
                               ? ort_value->Get<Tensor>().Shape().GetDims()
                               : ort_value->Get<SparseTensor>().DenseShape().GetDims();
#else
        ORT_ENFORCE(ort_value->IsTensor(), "Only OrtValues that are Tensors are supported in this build");
        const auto& dims = ort_value->Get<Tensor>().Shape().GetDims();
#endif

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
      .def("has_value", [](const OrtValue* ort_value) -> bool {
        return ort_value->IsAllocated();
      })
      .def("is_tensor", [](const OrtValue* ort_value) -> bool {
        return ort_value->IsTensor();
      })
      .def("is_sparse_tensor", [](const OrtValue* ort_value) -> bool {
        return ort_value->IsSparseTensor();
      })
      .def("is_tensor_sequence", [](const OrtValue* ort_value) -> bool {
        return ort_value->IsTensorSequence();
      })
      // Converts Tensor into a numpy array
      .def("numpy", [](const OrtValue* ml_value) -> py::object {
        ORT_ENFORCE(ml_value->IsTensor(), "Only OrtValues that are Tensors are convertible to Numpy objects");

        py::object obj;

#ifdef USE_CUDA
        GetPyObjFromTensor(ml_value->Get<Tensor>(), obj, nullptr, GetCudaToHostMemCpyFunction());
#elif USE_ROCM
  GetPyObjFromTensor(ml_value->Get<Tensor>(), obj, nullptr, GetRocmToHostMemCpyFunction());
#else
  GetPyObjFromTensor(ml_value->Get<Tensor>(), obj, nullptr, nullptr);
#endif
        return obj;
      })
#ifdef ENABLE_TRAINING
      .def("to_dlpack", [](OrtValue* ort_value) -> py::object {
        return py::reinterpret_steal<py::object>(ToDlpack(*ort_value));
      })
      .def_static("from_dlpack", [](py::object data, bool is_bool_tensor = false) {
        return FromDlpack(data.ptr(), is_bool_tensor);
      })
#endif
      ;
}

}  // namespace python
}  // namespace onnxruntime
