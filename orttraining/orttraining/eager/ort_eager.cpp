// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ort_eager_common.h"
#include "orttraining/core/framework/torch/dlpack_python.h"
#include <core/session/provider_bridge_ort.h>
#include "ort_backends.h"
#include "ort_log.h"
#include "ort_aten.h"
#include "ort_backends.h"
#include "orttraining/core/framework/ortmodule_graph_builder.h"
#include "ort_customops.h"
#include "torch/csrc/autograd/python_variable.h"
#include "core/framework/tensor.h"
#include "orttraining/python/orttraining_python_module_eager.h"

namespace onnxruntime{
namespace python{

using namespace onnxruntime::training;
using namespace torch_ort::eager;

static at::ScalarType aten_scalar_type_from_ort(
  onnxruntime::MLDataType dtype) {
  if (dtype == onnxruntime::DataTypeImpl::GetType<float>())
      return at::kFloat;
  else if (dtype == onnxruntime::DataTypeImpl::GetType<double>())
      return at::kDouble;
  else if (dtype == onnxruntime::DataTypeImpl::GetType<onnxruntime::MLFloat16>())
      return at::kHalf;
  else if (dtype == onnxruntime::DataTypeImpl::GetType<onnxruntime::BFloat16>())
      return at::kBFloat16;
  else if (dtype == onnxruntime::DataTypeImpl::GetType<int>())
      return at::kInt;
  else if (dtype == onnxruntime::DataTypeImpl::GetType<int16_t>())
      return at::kShort;
  else if (dtype == onnxruntime::DataTypeImpl::GetType<int64_t>())
      return at::kLong;
  else if (dtype == onnxruntime::DataTypeImpl::GetType<bool>())
      return at::kBool;
  else
      ORT_THROW("Unsupport aten scalar type: ", dtype);
}

OrtValue ORTTensor_toORTValue(const at::Tensor& data)
{
  return torch_ort::eager::create_ort_value(data);
}

at::Tensor OrtValue_To_ATen_Tensor(OrtValue& ortvalue)
{
  auto& ort_tensor = ortvalue.Get<Tensor>();
  size_t ort_device_idx = GetORTBackendsManager().GetOrtDeviceIndex(ort_tensor.Location());
  return torch_ort::eager::aten_tensor_from_ort(
    std::move(ortvalue),
    at::TensorOptions()
      .device(at::Device(at::DeviceType::ORT, ort_device_idx))
      .dtype(onnxruntime::python::aten_scalar_type_from_ort(ort_tensor.DataType())));
}

void addObjectMethodsForEager(py::module& m){
  ORT_LOG_INFO << "pybind11 module init";

  m.def(
    "device",
    [](int device_index) {
      return py::cast<py::object>(
        THPDevice_New(at::Device(at::DeviceType::ORT, device_index)));
    },
    py::arg("device_index") = 0);
  
  m.def("aten_ort_tensor_to_ort_value", [](at::Tensor data) {
    return ORTTensor_toORTValue(data);
  });
  m.def("to_aten_ort_device_tensor", [](OrtValue& ortvalue) {
    return OrtValue_To_ATen_Tensor(ortvalue);
  });

  m.def("set_device", [](size_t device_index, 
                                          const std::string& provider_type,
                                          const std::unordered_map<std::string, std::string>& arguments){
      auto status = GetORTBackendsManager().set_device(device_index, provider_type, arguments);
      if (!status.IsOK())
        throw std::runtime_error(status.ErrorMessage());
    });
  m.def("get_ort_device", [](size_t torch_device_index){
    return GetORTBackendsManager().GetOrtDeviceInfo(torch_device_index);
  });
  m.def("get_ort_device_provider_info", [](size_t torch_device_index){
    return GetORTBackendsManager().GetOrtDeviceProviderInfo(torch_device_index);
  });

  auto customop_module = m.def_submodule("custom_ops");
  torch_ort::eager::GenerateCustomOpsBindings(customop_module);
}

}
}
