// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/ml/label_encoder.h"
#include "core/providers/cuda/ml/label_encoder_impl.h"
#ifndef BUILD_CUDA_EP_AS_PLUGIN
#include "core/framework/tensorprotoutils.h"
#endif
#include "core/common/safeint.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace onnxruntime {
namespace cuda {

#ifndef BUILD_CUDA_EP_AS_PLUGIN
#ifdef SHARED_PROVIDER
using TensorProtoHolder = decltype(ONNX_NAMESPACE::TensorProto::Create());

static TensorProtoHolder CreateTensorProtoHolder() {
  return ONNX_NAMESPACE::TensorProto::Create();
}

static ONNX_NAMESPACE::TensorProto* GetTensorProto(TensorProtoHolder& holder) {
  return holder.get();
}
#else
using TensorProtoHolder = ONNX_NAMESPACE::TensorProto;

static TensorProtoHolder CreateTensorProtoHolder() {
  return {};
}

static ONNX_NAMESPACE::TensorProto* GetTensorProto(TensorProtoHolder& holder) {
  return &holder;
}
#endif
#endif

template <typename T>
static bool TryGetScalarTensorAttribute(const OpKernelInfo& info, const std::string& tensor_name,
                                        const std::string& attr_name, T& value) {
#ifdef BUILD_CUDA_EP_AS_PLUGIN
  // Plugin EP: use Ort C++ API to read tensor attributes.
  try {
    Ort::AllocatorWithDefaultOptions allocator;
    auto tensor_value = info.GetKernelInfo().GetTensorAttribute(tensor_name.c_str(), allocator);
    if (tensor_value.GetTensorTypeAndShapeInfo().GetElementCount() > 0) {
      value = *tensor_value.GetTensorData<T>();
      return true;
    }
  } catch (const Ort::Exception&) {
    // Tensor attribute not present, fall through to the caller's fallback path.
  }
#else
  // Non-plugin shared library EP: use TensorProto to read tensor attributes.
  auto attr_tensor_holder = CreateTensorProtoHolder();
  auto* attr_tensor_proto = GetTensorProto(attr_tensor_holder);
  auto result = info.GetAttr(tensor_name, attr_tensor_proto);
  if (result.IsOK() && utils::HasDataType(*attr_tensor_proto)) {
    result = utils::UnpackTensor<T>(*attr_tensor_proto, nullptr, 0, &value, 1);
    ORT_ENFORCE(result.IsOK(), "LabelEncoder could not unpack tensor attribute ", attr_name);
    return true;
  }
#endif  // BUILD_CUDA_EP_AS_PLUGIN

  return false;
}

// Helper to get attribute as vector from either list attributes or tensor attributes (for opset 4+).
template <typename T>
static std::vector<T> GetAttrOrTensor(const OpKernelInfo& info, const std::string& name,
                                      const std::string& tensor_name) {
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, int64_t>) {
    std::vector<T> attrs;
    if (info.GetAttrs<T>(name, attrs).IsOK()) {
      return attrs;
    }
  }
#ifdef BUILD_CUDA_EP_AS_PLUGIN
  // Plugin EP: use Ort C++ API to read tensor attributes.
  Ort::AllocatorWithDefaultOptions allocator;
  Ort::Value tensor_value{nullptr};
  try {
    tensor_value = info.GetKernelInfo().GetTensorAttribute(tensor_name.c_str(), allocator);
  } catch (const Ort::Exception&) {
    if (name.empty()) {
      ORT_THROW("LabelEncoder is missing attribute ", tensor_name);
    }
    ORT_THROW("LabelEncoder is missing attribute ", tensor_name, " or ", name);
  }
  size_t count = tensor_value.GetTensorTypeAndShapeInfo().GetElementCount();
  std::vector<T> out(count);
  std::memcpy(out.data(), tensor_value.GetTensorData<T>(), count * sizeof(T));
  return out;
#else
  // Non-plugin shared library EP: use TensorProto to read tensor attributes.
  auto attr_tensor_holder = CreateTensorProtoHolder();
  auto* attr_tensor_proto = GetTensorProto(attr_tensor_holder);
  auto result = info.GetAttr(tensor_name, attr_tensor_proto);
  if (name.empty()) {
    ORT_ENFORCE(result.IsOK(), "LabelEncoder is missing attribute ", tensor_name);
  } else {
    ORT_ENFORCE(result.IsOK(), "LabelEncoder is missing attribute ", tensor_name, " or ", name);
  }
  SafeInt<int64_t> element_count(1);
  for (auto dim : attr_tensor_proto->dims()) {
    element_count *= dim;
  }
  const SafeInt<size_t> tensor_size(element_count);
  std::vector<T> out(tensor_size);
  result = utils::UnpackTensor<T>(*attr_tensor_proto, nullptr, 0, out.data(), tensor_size);
  ORT_ENFORCE(result.IsOK(), "LabelEncoder could not unpack tensor attribute ", name);
  return out;
#endif  // BUILD_CUDA_EP_AS_PLUGIN
}

// Helper to get default value from default_tensor or a named attribute.
template <typename T>
static T GetDefaultValue(const OpKernelInfo& info, const std::string& attr_name, const T& backup) {
  T default_value;
  if (TryGetScalarTensorAttribute(info, "default_tensor", attr_name, default_value)) {
    return default_value;
  }

  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, int64_t>) {
    auto attr_result = info.GetAttr<T>(attr_name, &default_value);
    if (attr_result.IsOK()) {
      return default_value;
    }
  }
  return backup;
}

// Sort key-value pairs by key, deduplicate (first occurrence wins to match CPU
// emplace semantics), and handle NaN for floating-point types.
// Returns the index of the NaN key in the sorted arrays, or -1 if no NaN key.
template <typename TKey, typename TValue>
static int64_t SortKeysValues(std::vector<TKey>& keys, std::vector<TValue>& values) {
  // Create index array for sorting
  std::vector<size_t> indices(keys.size());
  std::iota(indices.begin(), indices.end(), 0);

  int64_t nan_key_index = -1;

  // Stable-sort indices by key value, placing NaN at the end.
  // Stable sort preserves the original order among equal keys so that the
  // first occurrence ends up first — matching CPU LabelEncoder's emplace().
  std::stable_sort(indices.begin(), indices.end(), [&keys](size_t a, size_t b) {
    if constexpr (std::is_floating_point_v<TKey>) {
      // NaN goes to end
      if (std::isnan(keys[a])) return false;
      if (std::isnan(keys[b])) return true;
    }
    return keys[a] < keys[b];
  });

  // Apply the sorted order and deduplicate, keeping only the first occurrence
  // of each key (which stable_sort guarantees is the original first).
  std::vector<TKey> sorted_keys;
  std::vector<TValue> sorted_values;
  sorted_keys.reserve(keys.size());
  sorted_values.reserve(values.size());

  for (size_t i = 0; i < indices.size(); ++i) {
    TKey k = keys[indices[i]];
    if (!sorted_keys.empty()) {
      if constexpr (std::is_floating_point_v<TKey>) {
        // Two NaNs are considered duplicates — keep only the first.
        if (std::isnan(k) && std::isnan(sorted_keys.back())) continue;
      }
      if (k == sorted_keys.back()) continue;
    }
    sorted_keys.push_back(k);
    sorted_values.push_back(values[indices[i]]);
  }

  // Find NaN key index (at the end if present)
  if constexpr (std::is_floating_point_v<TKey>) {
    if (!sorted_keys.empty() && std::isnan(sorted_keys.back())) {
      nan_key_index = static_cast<int64_t>(sorted_keys.size() - 1);
    }
  }

  keys = std::move(sorted_keys);
  values = std::move(sorted_values);
  return nan_key_index;
}

// Copy sorted key/value arrays to GPU memory.
template <typename TKey, typename TValue>
static void CopyToGpu(const OpKernelInfo& info,
                      const std::vector<TKey>& keys, const std::vector<TValue>& values,
                      IAllocatorUniquePtr<TKey>& keys_gpu, IAllocatorUniquePtr<TValue>& values_gpu) {
  auto alloc = info.GetAllocator(OrtMemTypeDefault);
  if (!keys.empty()) {
    keys_gpu = IAllocator::MakeUniquePtr<TKey>(alloc, keys.size());
    CUDA_CALL_THROW(cudaMemcpy(keys_gpu.get(), keys.data(),
                               SafeInt<size_t>(keys.size()) * sizeof(TKey),
                               cudaMemcpyHostToDevice));
    values_gpu = IAllocator::MakeUniquePtr<TValue>(alloc, values.size());
    CUDA_CALL_THROW(cudaMemcpy(values_gpu.get(), values.data(),
                               SafeInt<size_t>(values.size()) * sizeof(TValue),
                               cudaMemcpyHostToDevice));
  }
}

// ==============================
// CudaLabelEncoder (opset 2-3)
// ==============================

template <typename TKey, typename TValue>
CudaLabelEncoder<TKey, TValue>::CudaLabelEncoder(const OpKernelInfo& info)
    : CudaKernel(info), nan_key_index_(-1) {
  InitializeSomeFields(info);

  std::vector<TKey> keys;
  std::vector<TValue> values;

  ORT_THROW_IF_ERROR(info.GetAttrs<TKey>(key_field_name_, keys));
  ORT_THROW_IF_ERROR(info.GetAttrs<TValue>(value_field_name_, values));

  ORT_ENFORCE(keys.size() == values.size(),
              "The ", key_field_name_, " and ", value_field_name_,
              " attributes in LabelEncoder (name: ", info.node().Name(),
              ") must have the same length.");

  nan_key_index_ = SortKeysValues(keys, values);
  num_keys_ = static_cast<int64_t>(keys.size());
  CopyToGpu(info, keys, values, keys_gpu_, values_gpu_);
}

template <typename TKey, typename TValue>
Status CudaLabelEncoder<TKey, TValue>::ComputeInternal(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  const TensorShape& shape = X->Shape();
  auto* Y = context->Output(0, shape);

  int64_t num_elements = shape.Size();
  if (num_elements == 0) {
    return Status::OK();
  }

  LabelEncoderImpl(
      Stream(context),
      X->Data<TKey>(),
      Y->MutableData<TValue>(),
      num_elements,
      keys_gpu_.get(),
      values_gpu_.get(),
      num_keys_,
      default_value_,
      nan_key_index_);

  return Status::OK();
}

// Specializations for InitializeSomeFields

template <>
void CudaLabelEncoder<float, int64_t>::InitializeSomeFields(const OpKernelInfo& info) {
  key_field_name_ = "keys_floats";
  value_field_name_ = "values_int64s";
  info.GetAttrOrDefault<int64_t>("default_int64", &default_value_, static_cast<int64_t>(-1));
}

template <>
void CudaLabelEncoder<int64_t, float>::InitializeSomeFields(const OpKernelInfo& info) {
  key_field_name_ = "keys_int64s";
  value_field_name_ = "values_floats";
  info.GetAttrOrDefault<float>("default_float", &default_value_, -0.0f);
}

template <>
void CudaLabelEncoder<float, float>::InitializeSomeFields(const OpKernelInfo& info) {
  key_field_name_ = "keys_floats";
  value_field_name_ = "values_floats";
  info.GetAttrOrDefault<float>("default_float", &default_value_, -0.0f);
}

template <>
void CudaLabelEncoder<int64_t, int64_t>::InitializeSomeFields(const OpKernelInfo& info) {
  key_field_name_ = "keys_int64s";
  value_field_name_ = "values_int64s";
  info.GetAttrOrDefault<int64_t>("default_int64", &default_value_, static_cast<int64_t>(-1));
}

// ==============================
// CudaLabelEncoder_4 (opset 4+)
// ==============================

template <typename TKey, typename TValue>
CudaLabelEncoder_4<TKey, TValue>::CudaLabelEncoder_4(const OpKernelInfo& info)
    : CudaKernel(info), nan_key_index_(-1) {
  InitializeAttrFields(info);

  auto keys = GetAttrOrTensor<TKey>(info, key_field_name_, "keys_tensor");
  auto values = GetAttrOrTensor<TValue>(info, value_field_name_, "values_tensor");

  ORT_ENFORCE(keys.size() == values.size(), "Keys and values must have the same length.");

  nan_key_index_ = SortKeysValues(keys, values);
  num_keys_ = static_cast<int64_t>(keys.size());
  CopyToGpu(info, keys, values, keys_gpu_, values_gpu_);
}

template <typename TKey, typename TValue>
Status CudaLabelEncoder_4<TKey, TValue>::ComputeInternal(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  const TensorShape& shape = X->Shape();
  auto* Y = context->Output(0, shape);

  int64_t num_elements = shape.Size();
  if (num_elements == 0) {
    return Status::OK();
  }

  LabelEncoderImpl(
      Stream(context),
      X->Data<TKey>(),
      Y->MutableData<TValue>(),
      num_elements,
      keys_gpu_.get(),
      values_gpu_.get(),
      num_keys_,
      default_value_,
      nan_key_index_);

  return Status::OK();
}

// Specializations for InitializeAttrFields
// Note: For double types, key_field_name_ and value_field_name_ are intentionally
// left empty. When empty, GetAttrOrTensor skips the list-attribute path and reads
// directly from the tensor-based attributes (keys_tensor / values_tensor).
// This mirrors the CPU LabelEncoder_4 behavior for double types.

template <>
void CudaLabelEncoder_4<int64_t, int64_t>::InitializeAttrFields(const OpKernelInfo& info) {
  key_field_name_ = "keys_int64s";
  value_field_name_ = "values_int64s";
  default_value_ = GetDefaultValue(info, "default_int64", static_cast<int64_t>(-1));
}

template <>
void CudaLabelEncoder_4<int64_t, float>::InitializeAttrFields(const OpKernelInfo& info) {
  key_field_name_ = "keys_int64s";
  value_field_name_ = "values_floats";
  default_value_ = GetDefaultValue(info, "default_float", 0.f);
}

template <>
void CudaLabelEncoder_4<float, int64_t>::InitializeAttrFields(const OpKernelInfo& info) {
  key_field_name_ = "keys_floats";
  value_field_name_ = "values_int64s";
  default_value_ = GetDefaultValue(info, "default_int64", static_cast<int64_t>(-1));
}

template <>
void CudaLabelEncoder_4<float, float>::InitializeAttrFields(const OpKernelInfo& info) {
  key_field_name_ = "keys_floats";
  value_field_name_ = "values_floats";
  default_value_ = GetDefaultValue(info, "default_float", -0.f);
}

template <>
void CudaLabelEncoder_4<double, double>::InitializeAttrFields(const OpKernelInfo& info) {
  default_value_ = GetDefaultValue(info, "default_float", -0.);
}

template <>
void CudaLabelEncoder_4<double, int64_t>::InitializeAttrFields(const OpKernelInfo& info) {
  value_field_name_ = "values_int64s";
  default_value_ = GetDefaultValue(info, "default_int64", static_cast<int64_t>(-1));
}

template <>
void CudaLabelEncoder_4<int64_t, double>::InitializeAttrFields(const OpKernelInfo& info) {
  key_field_name_ = "keys_int64s";
  default_value_ = GetDefaultValue(info, "default_float", -0.);
}

// ==============================
// Kernel registrations
// ==============================

// Opset 2-3 registrations (numeric types only)

ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(
    LabelEncoder, kMLDomain, 2, 3, float_int64,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int64_t>()),
    CudaLabelEncoder<float, int64_t>);

ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(
    LabelEncoder, kMLDomain, 2, 3, int64_float,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<float>()),
    CudaLabelEncoder<int64_t, float>);

ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(
    LabelEncoder, kMLDomain, 2, 3, float_float,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<float>()),
    CudaLabelEncoder<float, float>);

ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(
    LabelEncoder, kMLDomain, 2, 3, int64_int64,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int64_t>()),
    CudaLabelEncoder<int64_t, int64_t>);

// Opset 4+ registrations (numeric types only)

ONNX_OPERATOR_TYPED_KERNEL_EX(
    LabelEncoder, kMLDomain, 4, int64_int64,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int64_t>()),
    CudaLabelEncoder_4<int64_t, int64_t>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    LabelEncoder, kMLDomain, 4, int64_float,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<float>()),
    CudaLabelEncoder_4<int64_t, float>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    LabelEncoder, kMLDomain, 4, float_int64,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int64_t>()),
    CudaLabelEncoder_4<float, int64_t>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    LabelEncoder, kMLDomain, 4, float_float,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<float>()),
    CudaLabelEncoder_4<float, float>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    LabelEncoder, kMLDomain, 4, double_double,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<double>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<double>()),
    CudaLabelEncoder_4<double, double>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    LabelEncoder, kMLDomain, 4, double_int64,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<double>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int64_t>()),
    CudaLabelEncoder_4<double, int64_t>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    LabelEncoder, kMLDomain, 4, int64_double,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<double>()),
    CudaLabelEncoder_4<int64_t, double>);

}  // namespace cuda
}  // namespace onnxruntime
