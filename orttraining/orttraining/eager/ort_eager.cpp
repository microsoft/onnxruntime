// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "python/onnxruntime_pybind_state_common.h"
#include "orttraining/core/framework/torch/dlpack_python.h"
#include <core/session/provider_bridge_ort.h>
#include "ort_backends.h"
#include "ort_log.h"
#include "ort_aten.h"
#include "ort_backends.h"
#include "orttraining/core/framework/ortmodule_graph_builder.h"
#include "ort_customops.h"
#include <torch/extension.h>
#include "torch/csrc/autograd/python_variable.h"

namespace onnxruntime{
namespace python{

using namespace onnxruntime::training;
using namespace torch_ort::eager;

py::object ORTTensor_toDLPack(const at::Tensor& data)
{
  OrtValue ort_value = torch_ort::eager::create_ort_value(data);
  return py::reinterpret_steal<py::object>(onnxruntime::training::framework::torch::ToDlpack(ort_value));
}

at::Tensor ORTTensor_FromDLPack(const py::object& dlpack_tensor)
{
  OrtValue ort_value = onnxruntime::training::framework::torch::FromDlpack(dlpack_tensor.ptr(), false);
  return torch_ort::eager::aten_tensor_from_ort(
    std::move(ort_value),
    at::TensorOptions()
      .device(at::Device(at::DeviceType::ORT, 0)));
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
  
  m.def("ort_to_dlpack", [](at::Tensor data) {
    return ORTTensor_toDLPack(data);
  });
  m.def("ort_from_dlpack", [](py::object dlpack_tensor) {
    return ORTTensor_FromDLPack(dlpack_tensor);
  });

  m.def("set_device", [](size_t device_index, 
                                          const std::string& provider_type,
                                          const std::unordered_map<std::string, std::string>& arguments){
      auto status = torch_ort::eager::GetORTBackendsManager().set_device(device_index, provider_type, arguments);
      if (!status.IsOK())
        throw std::runtime_error(status.ErrorMessage());
    });
  m.def("get_ort_device", [](size_t torch_device_index){
    return torch_ort::eager::GetORTBackendsManager().GetOrtDeviceInfo(torch_device_index);
  });

  auto customop_module = m.def_submodule("custom_ops");
  torch_ort::eager::GenerateCustomOpsBindings(customop_module);
}

}
}
