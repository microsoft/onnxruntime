#pragma once
#include "core/graph/model.h"
#include "core/providers/cpu/cpu_execution_provider.h"
namespace onnxruntime {
namespace training {
namespace api {

// static std::unique_ptr<Environment> env;
const std::vector<std::string> GRAD_SUFFIX{"_grad.accumulation.buffer", "_grad", "_grad.accumulation.out"};
const std::string MOMENT_1_SUFFIX{".exp_avg"};
const std::string MOMENT_2_SUFFIX{".exp_avg_sq"};
// TODO: don't hard code the state names, should get the state names according to the optimizer types.
const std::vector<std::string> MOMENT_STATE_NAMES{"momentum0", "momentum1"};

void GetGraphInputOutputNames(const Graph& graph,
                              std::vector<std::string>& input_names,
                              std::vector<std::string>& output_names);
bool GetParamNameFromSuffix(const std::string& name, const std::string& suffix, std::string& param_name);

bool GetParamNameFromGradient(const std::string& grad_name, std::string& param_name);

Status OrtValueLike(const SessionState& sess_state, const OrtValue& input_val, OrtValue& output_val);

template <typename T>
static void WarpInOrtValue(T value,
                           OrtValue* p_ortvalue,
                           AllocatorPtr alloc = nullptr) {
  static CPUExecutionProviderInfo info;
  static CPUExecutionProvider cpu_provider(info);
  static AllocatorPtr cpu_allocator = cpu_provider.GetAllocator(0, OrtMemTypeDefault);

  TensorShape shape({1});
  auto element_type = DataTypeImpl::GetType<T>();
  auto allocator = alloc ? alloc : cpu_allocator;
  auto p_tensor = std::make_unique<Tensor>(element_type, shape, allocator);
  memcpy(p_tensor->MutableDataRaw(), reinterpret_cast<void*>(&value), p_tensor->SizeInBytes());

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

template <typename T>
T GetValue(OrtValue& ort_value) {
  const Tensor& tensor = ort_value.Get<Tensor>();
  T val;
  if (DataTypeImpl::GetType<T>() == tensor.DataType()) {
    val = *(tensor.template Data<T>());
  } else {
    ORT_THROW("OrtValue data type not supported.");
  }
  return val;
}

}  // namespace api
}  // namespace training
}  // namespace onnxruntime