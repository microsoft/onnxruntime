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
#ifdef ENABLE_TRAINING
#include "core/dlpack/dlpack_converter.h"
#endif

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
#elif USE_DML
      // InputDeflist is null because OrtValue creation is not tied to a specific model
      // Likewise, there is no need to specify the name (as the name was previously used to lookup the def list)
      // TODO: Add check to ensure that string arrays are not passed - we currently don't support string tensors in DML
      CreateGenericMLValue(
        nullptr, GetDmlAllocator(device.Id()), "", array_on_cpu, ml_value.get(), true, false, CpuToDmlMemCpy);
#else
      throw std::runtime_error(
          "Can't allocate memory on the CUDA device using this package of OnnxRuntime. "
          "Please use the CUDA package of OnnxRuntime to use this feature.");
#endif
        } else if (device.Type() == OrtDevice::NPU) {
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
#elif USE_DML
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
        } else if (strcmp(GetDeviceName(device), DML) == 0) {
#if USE_DML
          allocator = GetDmlAllocator(device.Id());
#else
          throw std::runtime_error(
              "Can't allocate memory on the DirectML device using this package of OnnxRuntime. "
              "Please use the DirectML package of OnnxRuntime to use this feature.");
#endif
        } else {
          throw std::runtime_error("Unsupported device: Cannot place the OrtValue on this device");
        }

        auto ml_value = std::make_unique<OrtValue>();
        auto ml_type = NumpyTypeToOnnxRuntimeTensorType(type_num);
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
      .def(
          "element_type", [](const OrtValue* ort_value) -> int32_t {
            return GetTensorProtoType(*ort_value);
          },
          "Returns an integer equal to the ONNX tensor proto type of the tensor or sequence. "
          "This integer is one type defined by ONNX TensorProto_DataType "
          "(such as onnx.TensorProto.FLOAT)."
          "Raises an exception in any other case.")
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
#elif USE_CANN
        GetPyObjFromTensor(ml_value->Get<Tensor>(), obj, nullptr, GetCannToHostMemCpyFunction());
#elif USE_DML
        GetPyObjFromTensor(ml_value->Get<Tensor>(), obj, nullptr, GetDmlToHostMemCpyFunction());
#else
        GetPyObjFromTensor(ml_value->Get<Tensor>(), obj, nullptr, nullptr);
#endif
        return obj;
      })
#ifdef ENABLE_TRAINING
      .def(
          "to_dlpack", [](OrtValue* ort_value) -> py::object {
            return py::reinterpret_steal<py::object>(ToDlpack(*ort_value));
          },
          "Returns a DLPack representing the tensor. This method does not copy the pointer shape, "
          "instead, it copies the pointer value. The OrtValue must be persist until the dlpack structure "
          "is consumed.")
      .def_static(
          "from_dlpack", [](py::object data, bool is_bool_tensor) {
            return FromDlpack(data.ptr(), is_bool_tensor);
          },
          py::arg("data"), py::arg("is_bool_tensor") = false, "Converts a tensor from a external library into an OrtValue by means of the __dlpack__ protocol.")
      .def(
          "__dlpack__", [](OrtValue* ort_value, py::object /* stream */) -> py::object {
            return py::reinterpret_steal<py::object>(ToDlpack(*ort_value));
          },
          py::arg("stream") = py::none(),
          "Returns a DLPack representing the tensor (part of __dlpack__ protocol). "
          "This method does not copy the pointer shape, instead, it copies the pointer value. "
          "The OrtValue must persist until the dlpack structure is consumed.")
      .def(
          "__dlpack_device__", [](const OrtValue* ort_value) -> py::tuple {
            ORT_ENFORCE(ort_value->IsTensor(), "Only tensor type OrtValues are supported");
            const onnxruntime::Tensor& tensor = ort_value->Get<Tensor>();
            DLDevice device = onnxruntime::dlpack::GetDlpackDevice(*ort_value, tensor.Location().device.Id());
            return py::make_tuple(static_cast<int>(device.device_type), device.device_id);
          },
          "Returns a tuple of integers, (device, device index) (part of __dlpack__ protocol).")
#endif
      ;

  py::class_<std::vector<OrtValue>>(m, "OrtValueVector")
      .def(py::init<>())
      .def("push_back", [](std::vector<OrtValue>* v, const OrtValue& ortvalue) {
        v->push_back(ortvalue);
      })
#ifdef ENABLE_TRAINING
      .def(
          "push_back", [](std::vector<OrtValue>* v, py::object dlpack_tensor, const bool is_bool_tensor) {
            v->push_back(FromDlpack(dlpack_tensor.ptr(), is_bool_tensor));
          },
          "Add a new OrtValue after being ownership was transferred from the DLPack structure.", py::arg("dlpack_tensor"), py::arg("is_bool_tensor") = false)
      .def(
          "push_back_batch", [](std::vector<OrtValue>* v, std::vector<py::object>& torch_tensors, std::vector<int64_t>& data_ptrs, std::vector<py::object>& element_types, const std::vector<std::vector<int64_t>>& shapes, const std::vector<OrtDevice>& devices) {
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
              OrtMemoryInfo info(GetDeviceName(device), OrtDeviceAllocator, device, device.Id());
              OrtValue ml_value;
              Tensor::InitOrtValue(ml_type, gsl::make_span(shape), reinterpret_cast<void*>(data_ptr), info, ml_value);
              v->push_back(ml_value);
            }
          },
          "Add a batch of OrtValue's by wrapping PyTorch tensors.")
#endif
      .def("reserve", [](std::vector<OrtValue>* v, const size_t len) { v->reserve(len); })
      .def("shrink_to_fit", [](std::vector<OrtValue>* v) { v->shrink_to_fit(); })
      .def("__len__", [](const std::vector<OrtValue>& v) { return v.size(); })
      .def(
          "__iter__", [](const std::vector<OrtValue>& v) {
            return py::make_iterator(v.cbegin(), v.cend());
          },
          py::keep_alive<0, 1>())
      .def("__getitem__", [](const std::vector<OrtValue>& v, const size_t idx) {
        return v.at(idx);
      })
      .def(
          "bool_tensor_indices", [](std::vector<OrtValue>* v) -> std::vector<int64_t> {
            std::vector<int64_t> indices;
            for (size_t i = 0; i < v->size(); ++i) {
              if (GetTensorProtoType((*v)[i]) == ONNX_NAMESPACE::TensorProto_DataType_BOOL) {
                indices.push_back(static_cast<int64_t>(i));
              }
            }
            return indices;
          },
          "Returns the indices of every boolean tensor in this vector of OrtValue. "
          "In case of a boolean tensor, method to_dlpacks returns a uint8 tensor instead of a boolean tensor. "
          "If torch consumes the dlpack structure, `.to(torch.bool)` must be applied to the torch tensor "
          "to get a boolean tensor.")
#ifdef ENABLE_TRAINING
      .def("dlpack_at", [](std::vector<OrtValue>* v, const size_t idx) {
        return py::reinterpret_steal<py::object>(ToDlpack(v->at(idx)));
      })
#endif
      .def(
          "element_type_at", [](std::vector<OrtValue>* v, const size_t idx) -> int32_t {
            return GetTensorProtoType(v->at(idx));
          },
          "Returns an integer equal to the ONNX proto type of the tensor at position i. "
          "This integer is one type defined by ONNX TensorProto_DataType "
          "(such as onnx.TensorProto.FLOAT)."
          "Raises an exception in any other case.",
          py::arg("idx"))
#ifdef ENABLE_TRAINING
      .def(
          "to_dlpacks", [](const std::vector<OrtValue>& v, py::object to_tensor) -> py::list {
            if (v.size() == 0)
              return py::list();

            py::list list_dlpacks;
            PyObject* obj;

            py::gil_scoped_acquire acquire;

            if (to_tensor.is_none()) {
              DLManagedTensor* dlmanaged_tensor;

              for (auto it : v) {
                dlmanaged_tensor = dlpack::OrtValueToDlpack(it);
                py::capsule capsule(dlmanaged_tensor, "dltensor", DlpackCapsuleDestructor);
                list_dlpacks.append(capsule);
              }
            } else {
              DLManagedTensor* dlmanaged_tensor;
              PyObject* capsule = NULL;
              PyObject* handle = to_tensor.ptr();

              for (auto it : v) {
                // A new instance of dlpack needs to be created. The object which consumes it
                // is responsible for its deletion.
                dlmanaged_tensor = dlpack::OrtValueToDlpack(it);
                if (capsule == NULL) {
                  capsule = PyCapsule_New(dlmanaged_tensor, "dltensor", NULL);
                  if (capsule == NULL)
                    throw std::runtime_error("Unexpected error: empty capsule returned.");
                } else {
                  // The same capsule is reused but FromDLPack rename the capsule into used_dltensor.
                  PyCapsule_SetName(capsule, "dltensor");
                  PyCapsule_SetPointer(capsule, dlmanaged_tensor);
                }
                obj = PyObject_CallFunctionObjArgs(handle, capsule, NULL);
                if (obj == NULL)
                  throw std::runtime_error("to_tensor returned a null pointer. This may be caused by the data conversion.");
                list_dlpacks.append(obj);
                Py_DECREF(obj);
              }
              if (capsule != NULL) {
                // This test is never wrong because v is not empty if the execution goes through that path.
                // If not present, Guardian detects a potential failure.
                Py_DECREF(capsule);
              }
            }
            return list_dlpacks;
          },
          R"pbdoc(Converts all OrtValue into tensors through DLPack protocol, the method creates
a DLPack structure for every tensors, then calls python function `to_tensor` to a new object
consuming the DLPack structure or return a list of capsule if this function is None.

:param to_tensor: this function takes a capsule holding a pointer onto a DLPack structure and returns
    a new tensor which becomes the new owner of the data. This function takes one python object and
    returns a new python object. It fits the same signature as `torch.utils.from_dlpack`,
    if None, the method returns a capsule for every new DLPack structure.
:return: a list containing the new tensors or a the new capsules if *to_tensor* is None

This method is used to replace `tuple(torch._C._from_dlpack(ov.to_dlpack()) for ov in ort_values)`
by a faster instruction `tuple(ort_values.to_dlpack(torch._C._from_dlpack))`. This loop
is difficult to parallelize as it goes through the GIL many times.
It creates many tensors acquiring ownership of existing OrtValue.
This method saves one object creation and an C++ allocation
for every transferred tensor.
)pbdoc",
          py::arg("to_tensor"))
#endif
      ;

#ifdef ENABLE_TRAINING
  m.def(
      "is_dlpack_uint8_tensor", [](py::capsule cap) -> bool {
        // case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
        // dtype.code = DLDataTypeCode::kDLUInt;
        // dtype.bits = sizeof(bool);
        DLManagedTensor* dlmanaged_tensor = (DLManagedTensor*)cap.get_pointer();
        return dlmanaged_tensor->dl_tensor.dtype.code == DLDataTypeCode::kDLUInt && dlmanaged_tensor->dl_tensor.dtype.bits == 8;
      },
      "Tells if a DLPack structure is a uint8 tensor.\n"
      ".. note::\n"
      "    Boolean tensors are also uint8 tensor once converted with DLPack protocol.");
#endif
}

}  // namespace python
}  // namespace onnxruntime
