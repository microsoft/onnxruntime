#pragma once
#include "core/graph/model.h"
#include "core/providers/cpu/cpu_execution_provider.h"
namespace onnxruntime {
namespace training {
namespace api {

// static std::unique_ptr<Environment> env;
const std::vector<std::string> GRAD_SUFFIX{"_grad.accumulation.buffer", "_grad"};
const std::string MOMENT_1{".exp_avg"};
const std::string MOMENT_2{".exp_avg_sq"};

void GetGraphInputOutputNames(const Graph& graph,
                              std::vector<std::string>& input_names,
                              std::vector<std::string>& output_names);
bool GetParamNameFromSuffix(const std::string& name, const std::string& suffix, std::string& param_name);

bool GetParamNameFromGradient(const std::string& grad_name, std::string& param_name);

Status OrtValueLike(const SessionState& sess_state, const OrtValue& input_val, OrtValue& output_val);

template <typename T>
static void CreateInputOrtValue(T value,
                                OrtValue* p_ortvalue,
                                AllocatorPtr alloc = nullptr) {
  static CPUExecutionProviderInfo info;
  static CPUExecutionProvider cpu_provider(info);
  static AllocatorPtr cpu_allocator = cpu_provider.GetAllocator(0, OrtMemTypeDefault);

  TensorShape shape({1});
  auto element_type = DataTypeImpl::GetType<T>();
  auto allocator = alloc ? alloc : cpu_allocator;
  auto p_tensor = std::make_unique<Tensor>(element_type, shape, allocator);
  memcpy(p_tensor->MutableDataRaw(), reinterpret_cast<void *>(&value), p_tensor->SizeInBytes());

  p_ortvalue->Init(p_tensor.release(),
                   DataTypeImpl::GetType<Tensor>(),
                   DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
}

template <typename T>
static void CreateInputOrtValue(gsl::span<const int64_t> dims,
                                const std::vector<T>& value,
                                OrtValue* p_ortvalue,
                                AllocatorPtr alloc = nullptr) {
  static CPUExecutionProviderInfo info;
  static CPUExecutionProvider cpu_provider(info);
  static AllocatorPtr cpu_allocator = cpu_provider.GetAllocator(0, OrtMemTypeDefault);

  TensorShape shape(dims);
  assert(shape.Size() == static_cast<int64_t>(value.size()));
  auto element_type = DataTypeImpl::GetType<T>();
  auto allocator = alloc ? alloc : cpu_allocator;
  auto p_tensor = std::make_unique<Tensor>(element_type, shape, allocator);

  if (value.size() > 0) {
    memcpy(p_tensor->MutableDataRaw(), value.data(), p_tensor->SizeInBytes());
  }

  p_ortvalue->Init(p_tensor.release(),
                   DataTypeImpl::GetType<Tensor>(),
                   DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
}

}  // namespace api
}  // namespace training
}  // namespace onnxruntime