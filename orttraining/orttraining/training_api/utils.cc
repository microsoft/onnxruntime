#include "core/graph/model.h"
#include "core/framework/session_state.h"
#include "core/framework/ort_value.h"
#include "core/framework/tensor.h"
#include "core/framework/allocator.h"

#include "orttraining/training_api/include/utils.h"

namespace onnxruntime {
namespace training {
namespace api {

void GetGraphInputOutputNames(const Graph& graph,
                              std::vector<std::string>& input_names,
                              std::vector<std::string>& output_names) {
  auto inputs = graph.GetInputs();
  auto outputs = graph.GetOutputs();

  auto get_names = [&](const std::vector<const NodeArg*>& node_args, std::vector<std::string>& names) {
    for (const auto* arg : node_args) {
      names.push_back(arg->Name());
    }
  };

  get_names(inputs, input_names);
  get_names(outputs, output_names);
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
  AllocatorPtr allocator = sess_state.GetAllocator(param_tensor.Location());
  // AllocatorPtr allocator = GetAllocator(param_tensor.Location());

  auto element_type = param_tensor.DataType();
  auto p_tensor = std::make_unique<Tensor>(element_type, shape, allocator);

  // TODO: handle CUDA memset
  memset(p_tensor->MutableDataRaw(), 0, p_tensor->SizeInBytes());
  output_val.Init(p_tensor.release(),
                  DataTypeImpl::GetType<Tensor>(),
                  DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
  return Status::OK();
}

}  // namespace api
}  // namespace training
}  // namespace onnxruntime
