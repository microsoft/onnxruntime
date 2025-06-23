#include "python/onnxruntime_pybind_exceptions.h"
#include "python/onnxruntime_pybind_mlvalue.h"
#include "python/onnxruntime_pybind_state_common.h"

#define NO_IMPORT_ARRAY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL onnxruntime_python_ARRAY_API
#include "python/numpy_helper.h"
#include "core/framework/ort_value.h"
#include "core/framework/tensor.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/TensorSeq.h"
#include "core/session/IOBinding.h"

namespace onnxruntime {
namespace python {

// Use the nanobind namespace
namespace nb = nanobind;

namespace {
// This helper function does not need changes as it uses core ONNX Runtime and C++ types.
void BindOutput(SessionIOBinding* io_binding, const std::string& name, const OrtDevice& device,
                MLDataType element_type, const std::vector<int64_t>& shape, int64_t data_ptr) {
  ORT_ENFORCE(data_ptr != 0, "Pointer to data memory is not valid");
  InferenceSession* sess = io_binding->GetInferenceSession();
  auto px = sess->GetModelOutputs();
  if (!px.first.IsOK() || !px.second) {
    throw std::runtime_error("Either failed to get model inputs from the session object or the input def list was null");
  }

  const auto& def_list = *px.second;
  onnx::TypeProto type_proto;
  if (!CheckIfTensor(def_list, name, type_proto)) {
    throw std::runtime_error("Only binding Tensors is currently supported");
  }

  ORT_ENFORCE(utils::HasTensorType(type_proto) && utils::HasElemType(type_proto.tensor_type()));
  if (type_proto.tensor_type().elem_type() == onnx::TensorProto::STRING) {
    throw std::runtime_error("Only binding non-string Tensors is currently supported");
  }

  OrtValue ml_value;
  OrtMemoryInfo info(GetDeviceName(device), OrtDeviceAllocator, device);
  Tensor::InitOrtValue(element_type, gsl::make_span(shape), reinterpret_cast<void*>(data_ptr), info, ml_value);

  auto status = io_binding->Get()->BindOutput(name, ml_value);
  if (!status.IsOK()) {
    throw std::runtime_error("Error when binding output: " + status.ErrorMessage());
  }
}
}  // namespace

void addIoBindingMethods(nanobind::module_& m) {
  // Use nb::class_
  nb::class_<SessionIOBinding> session_io_binding(m, "SessionIOBinding");
  session_io_binding
      .def("__init__",
           [](SessionIOBinding* t, PyInferenceSession* sess) {
             new (t) SessionIOBinding(sess->GetSessionHandle());
           })
      .def("bind_input", [](SessionIOBinding* io_binding, const std::string& name, nb::object& arr_on_cpu) -> void {
        InferenceSession* sess = io_binding->GetInferenceSession();
        auto px = sess->GetModelInputs();
        if (!px.first.IsOK() || !px.second) {
          throw std::runtime_error("Either failed to get model inputs from the session object or the input def list was null");
        }

        const auto& def_list = *px.second;
        onnx::TypeProto type_proto;
        if (!CheckIfTensor(def_list, name, type_proto)) {
          throw std::runtime_error("Only binding Tensors is currently supported");
        }

        ORT_ENFORCE(utils::HasTensorType(type_proto) && utils::HasElemType(type_proto.tensor_type()));
        if (type_proto.tensor_type().elem_type() == onnx::TensorProto::STRING) {
          throw std::runtime_error("Only binding non-string Tensors is currently supported");
        }

        OrtValue ml_value;
        CreateGenericMLValue(px.second, GetAllocator(), name, arr_on_cpu, &ml_value, true);

        auto status = io_binding->Get()->BindInput(name, ml_value);
        if (!status.IsOK()) {
          throw std::runtime_error("Error when bind input: " + status.ErrorMessage());
        }
      })
      .def("bind_input", [](SessionIOBinding* io_binding, const std::string& name, const OrtDevice& device, int32_t element_type, const std::vector<int64_t>& shape, int64_t data_ptr) -> void {
        auto ml_type = OnnxTypeToOnnxRuntimeTensorType(element_type);
        OrtValue ml_value;
        OrtMemoryInfo info(GetDeviceName(device), OrtDeviceAllocator, device);
        Tensor::InitOrtValue(ml_type, gsl::make_span(shape), reinterpret_cast<void*>(data_ptr), info, ml_value);

        auto status = io_binding->Get()->BindInput(name, ml_value);
        if (!status.IsOK()) {
          throw std::runtime_error("Error when binding input: " + status.ErrorMessage());
        }
      })
      .def("bind_input", [](SessionIOBinding* io_binding, const std::string& name, const OrtDevice& device, nb::object& element_type, const std::vector<int64_t>& shape, int64_t data_ptr) -> void {
        PyArray_Descr* dtype;
        if (!PyArray_DescrConverter(element_type.ptr(), &dtype)) {
          throw std::runtime_error("Not a valid numpy type");
        }
        int type_num = dtype->type_num;
        Py_DECREF(dtype);

        OrtMemoryInfo info(GetDeviceName(device), OrtDeviceAllocator, device);
        auto ml_type = NumpyTypeToOnnxRuntimeTensorType(type_num);
        OrtValue ml_value;
        Tensor::InitOrtValue(ml_type, gsl::make_span(shape), reinterpret_cast<void*>(data_ptr), info, ml_value);

        auto status = io_binding->Get()->BindInput(name, ml_value);
        if (!status.IsOK()) {
          throw std::runtime_error("Error when binding input: " + status.ErrorMessage());
        }
      })
      .def("bind_ortvalue_input", [](SessionIOBinding* io_binding, const std::string& name, const OrtValue& ml_value) -> void {
        auto status = io_binding->Get()->BindInput(name, ml_value);
        if (!status.IsOK()) {
          throw std::runtime_error("Error when binding input: " + status.ErrorMessage());
        }
      })
      .def("synchronize_inputs", [](SessionIOBinding* io_binding) -> void {
        auto status = io_binding->Get()->SynchronizeInputs();
        if (!status.IsOK()) {
          throw std::runtime_error("Error when synchronizing bound inputs: " + status.ErrorMessage());
        }
      })
      .def("bind_output", [](SessionIOBinding* io_binding, const std::string& name, const OrtDevice& device, int32_t element_type, const std::vector<int64_t>& shape, int64_t data_ptr) -> void {
        MLDataType ml_type = OnnxTypeToOnnxRuntimeTensorType(element_type);
        BindOutput(io_binding, name, device, ml_type, shape, data_ptr);
      })
      .def("bind_output", [](SessionIOBinding* io_binding, const std::string& name, const OrtDevice& device, nb::object& element_type, const std::vector<int64_t>& shape, int64_t data_ptr) -> void {
        PyArray_Descr* dtype;
        if (!PyArray_DescrConverter(element_type.ptr(), &dtype)) {
          throw std::runtime_error("Not a valid numpy type");
        }
        int type_num = dtype->type_num;
        Py_DECREF(dtype);

        auto ml_type = NumpyTypeToOnnxRuntimeTensorType(type_num);
        BindOutput(io_binding, name, device, ml_type, shape, data_ptr);
      })
      .def("bind_output", [](SessionIOBinding* io_binding, const std::string& name, const OrtDevice& device) -> void {
        auto status = io_binding->Get()->BindOutput(name, device);
        if (!status.IsOK()) {
          throw std::runtime_error("Error when binding output: " + status.ErrorMessage());
        }
      })
      .def("bind_ortvalue_output", [](SessionIOBinding* io_binding, const std::string& name, const OrtValue& ml_value) -> void {
        auto status = io_binding->Get()->BindOutput(name, ml_value);
        if (!status.IsOK()) {
          throw std::runtime_error("Error when binding output: " + status.ErrorMessage());
        }
      })
      .def("synchronize_outputs", [](SessionIOBinding* io_binding) -> void {
        auto status = io_binding->Get()->SynchronizeOutputs();
        if (!status.IsOK()) {
          throw std::runtime_error("Error when synchronizing bound outputs: " + status.ErrorMessage());
        }
      })
      .def("clear_binding_inputs", [](SessionIOBinding* io_binding) -> void {
        io_binding->Get()->ClearInputs();
      })
      .def("clear_binding_outputs", [](SessionIOBinding* io_binding) -> void {
        io_binding->Get()->ClearOutputs();
      })
      // Use nb::rv_policy
      .def("get_outputs", [](const SessionIOBinding* io_binding) -> const std::vector<OrtValue>& { return io_binding->Get()->GetOutputs(); }, nb::rv_policy::reference_internal)
      .def("copy_outputs_to_cpu", [](const SessionIOBinding* io_binding) -> nb::list {
        const std::vector<OrtValue>& outputs = io_binding->Get()->GetOutputs();
        size_t pos = 0;
        const auto& dtm = io_binding->GetInferenceSession()->GetDataTransferManager();
        nb::list result;
        for (const auto& ort_value : outputs) {
          if (ort_value.IsTensor()) {
            // We make a copy of the tensor to CPU even if it is already on CPU
            // as the function name implies using DataTransferManager.
            nb::ndarray arr = PrimitiveTensorToNumpyFromDevice(ort_value, &dtm);
            result.append(arr);
          } else if (ort_value.IsSparseTensor()) {
            result.append(GetPyObjectFromSparseTensor(pos, ort_value, &dtm));
          } else {
            result.append(AddNonTensorAsPyObj(ort_value, &dtm, nullptr));
          }
          ++pos;
        }
        return result; });
}

}  // namespace python
}  // namespace onnxruntime