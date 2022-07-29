// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/model.h"
#include "core/framework/session_state.h"
#include "core/framework/ort_value.h"
#include "core/framework/tensor.h"
#include "core/framework/allocator.h"

#include "orttraining/training_api/include/utils.h"

namespace onnxruntime {
namespace training {
namespace api {
namespace utils {

// TODO: consolidate the gradient names with frontend tooling
const std::vector<std::string> GRAD_SUFFIX{"_grad.accumulation.buffer", "_grad", "_grad.accumulation.out"};

void GetGraphInputOutputNames(const std::unique_ptr<onnxruntime::InferenceSession>& session_object,
                              std::vector<std::string>& input_names,
                              std::vector<std::string>& output_names) {
  auto get_names = [](const std::vector<const NodeArg*>* node_args, std::vector<std::string>& names) {
    ORT_ENFORCE(nullptr != node_args);
    for (const auto* arg : *node_args) {
      names.push_back(arg->Name());
    }
  };
  const auto& retval_input = session_object->GetModelInputs();
  ORT_ENFORCE(retval_input.first.IsOK());
  get_names(retval_input.second, input_names);
  const auto& retval_output = session_object->GetModelOutputs();
  ORT_ENFORCE(retval_output.first.IsOK());
  get_names(retval_output.second, output_names);
}

bool GetParamNameFromSuffix(const std::string& name, const std::string& suffix, std::string& param_name) {
  bool endswith = std::equal(suffix.rbegin(), suffix.rend(), name.rbegin());
  if (endswith) {
    param_name = name.substr(0, name.length() - suffix.length());
    return true;
  } else {
    param_name = "";
    return false;
  }
}

bool GetParamNameFromGradient(const std::string& grad_name, std::string& param_name) {
  for (auto& suffix : GRAD_SUFFIX) {
    if (GetParamNameFromSuffix(grad_name, suffix, param_name)) {
      return true;
    }
  }
  return false;
}

Status OrtValueLike(const SessionState& sess_state, const OrtValue& input_val, OrtValue& output_val) {
  const auto& param_tensor = input_val.template Get<Tensor>();
  const TensorShape& shape = param_tensor.Shape();
  auto& tensor_location = param_tensor.Location();
  AllocatorPtr allocator = sess_state.GetAllocator(tensor_location);

  auto element_type = param_tensor.DataType();
  auto p_tensor = std::make_unique<Tensor>(element_type, shape, allocator);

  if (tensor_location.device.Type() == OrtDevice::CPU ||
      tensor_location.mem_type == OrtMemTypeCPUInput ||
      tensor_location.mem_type == OrtMemTypeCPUOutput) {
    memset(p_tensor->MutableDataRaw(), 0, p_tensor->SizeInBytes());
  } else if (tensor_location.device.Type() == OrtDevice::GPU) {
    // Use a tensor on cpu and copy it over to the desired device using
    // the data transfer manager.
    AllocatorPtr cpu_allocator = sess_state.GetAllocator(OrtDevice());
    auto p_cpu_tensor = std::make_unique<Tensor>(element_type, shape, cpu_allocator);
    memset(p_cpu_tensor->MutableDataRaw(), 0, p_cpu_tensor->SizeInBytes());
    // No need to free the cpu buffer, tensor destructor takes care of it using the cpu_allocator
    ORT_THROW_IF_ERROR(sess_state.GetDataTransferMgr().CopyTensor(*p_cpu_tensor, *p_tensor));
  } else {
    ORT_THROW("Cannot create tensor on device ", tensor_location.device.Type());
  }

  output_val.Init(p_tensor.release(),
                  DataTypeImpl::GetType<Tensor>(),
                  DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
  return Status::OK();
}

}  // namespace utils
}  // namespace api
}  // namespace training
}  // namespace onnxruntime
