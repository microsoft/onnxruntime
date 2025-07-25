// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"

#include <functional>
#include <memory>
#include <string>
#include <gsl/gsl>
#include "core/framework/tensorprotoutils.h"
#include "core/framework/tensor_external_data_info.h"
#include "core/platform/env.h"
#include "core/mlas/inc/mlas.h"
#include "core/common/cpuid_info.h"

namespace onnxruntime {

static inline Tensor* GetTensor(OrtValue& ort_value) {
  return ort_value.GetMutable<Tensor>();
}

Initializer::Initializer(ONNX_NAMESPACE::TensorProto_DataType data_type,
                         std::string_view name,
                         gsl::span<const int64_t> dims) : name_(name) {
  auto tensor = Tensor(DataTypeImpl::TensorTypeFromONNXEnum(data_type)->GetElementType(), dims,
                       CPUAllocator::DefaultInstance());

  if (!tensor.IsDataTypeString()) {
    memset(tensor.MutableDataRaw(), 0, tensor.SizeInBytes());
  }

  Tensor::InitOrtValue(std::move(tensor), ort_value_);
  data_ = GetTensor(ort_value_);
}

Initializer::Initializer(const ONNX_NAMESPACE::TensorProto& tensor_proto, const std::filesystem::path& model_path) {
  ORT_ENFORCE(utils::HasName(tensor_proto), "Initializer must have a name");
  name_ = tensor_proto.name();

#if !defined(__wasm__)
  // using full filepath is required by utils::TensorProtoToTensor(). One exception is WebAssembly platform, where
  // external data is not loaded from real file system.
  if (utils::HasExternalData(tensor_proto) && !utils::HasExternalDataInMemory(tensor_proto)) {
    ORT_ENFORCE(!model_path.empty(),
                "model_path must not be empty. Ensure that a path is provided when the model is created or loaded.");
  }
#endif

  Tensor tensor;
  // This creates copy of the data so clients can mutate
  ORT_THROW_IF_ERROR(utils::CreateTensorFromTensorProto(Env::Default(), model_path, tensor_proto, tensor));
  Tensor::InitOrtValue(std::move(tensor), ort_value_);
  data_ = GetTensor(ort_value_);
}

Initializer::Initializer(const Graph& graph, const ONNX_NAMESPACE::TensorProto& tensor_proto,
                         const std::filesystem::path& model_path, bool check_outer_scope) {
  ORT_ENFORCE(utils::HasName(tensor_proto), "Initializer must have a name");
  name_ = tensor_proto.name();

  // Check if the data is in memory. This does not mean, though, that the data is in the ort_value
  if (utils::HasExternalDataInMemory(tensor_proto)) {
    OrtValue ort_value;
    if (graph.GetOrtValueInitializer(name_, ort_value, check_outer_scope)) {
      const auto& src_tensor = ort_value.Get<Tensor>();
      // We need to make a copy of the data to ensure that the original data is not mutated
      // This is generally inline with TensorProtoToTensor() behavior which copies data from
      // TensorProto to Tensor.
      Tensor initializer{src_tensor.DataType(), src_tensor.Shape(), CPUAllocator::DefaultInstance()};
      utils::MakeCpuTensorCopy(src_tensor, initializer);
      Tensor::InitOrtValue(std::move(initializer), ort_value_);
      data_ = GetTensor(ort_value_);
      return;
    }
#if !defined(__wasm__)
    ORT_ENFORCE(!model_path.empty(),
                "model_path must not be empty. Ensure that a path is provided when the model is created or loaded.");
#endif
  }

  Tensor tensor;
  // Creates a copy of the data from tensor_proto
  ORT_THROW_IF_ERROR(utils::CreateTensorFromTensorProto(Env::Default(), model_path, tensor_proto, tensor));
  Tensor::InitOrtValue(std::move(tensor), ort_value_);
  data_ = GetTensor(ort_value_);
}

Initializer::~Initializer() = default;

void Initializer::ToProtoWithOrtValue(ONNX_NAMESPACE::TensorProto& tensor_proto, OrtValue& ort_value) const {
  constexpr const bool use_tensor_buffer_true = true;
  tensor_proto = utils::TensorToTensorProto(*data_, name_, use_tensor_buffer_true);
  if (utils::HasExternalDataInMemory(tensor_proto)) {
    ort_value = ort_value_;
  } else {
    ort_value = OrtValue();
  }
}

#if !defined(ORT_EXTENDED_MINIMAL_BUILD)
namespace {
template <typename T>
struct ToFp16;

template <>
struct ToFp16<MLFloat16> {
  uint16_t operator()(const MLFloat16& fl) const {
    return fl.val;
  }
};

template <>
struct ToFp16<float> {
  uint16_t operator()(float f) const {
    return MLFloat16(f).val;
  }
};

template <>
struct ToFp16<double> {
  uint16_t operator()(double d) const {
    // The same code as in Eigen. We assume the loss of precision will occur
    // hence static_cast
    return MLFloat16(static_cast<float>(d)).val;
  }
};

template <typename T>
struct TensorToProtoFP16 {
  void operator()(const Tensor& data, ONNX_NAMESPACE::TensorProto& proto) const {
    ToFp16<T> to_fp16;
    auto span = data.DataAsSpan<T>();
    for (const auto& v : span) {
      proto.add_int32_data(to_fp16(v));
    }
  }
};

template <typename T>
struct TensorToFP16 {
  void operator()(const Tensor& data, Tensor& dst) const {
    ToFp16<T> to_fp16;
    auto span = data.DataAsSpan<T>();
    auto* dst_data = dst.MutableData<MLFloat16>();
    for (const auto& v : span) {
      *dst_data++ = MLFloat16::FromBits(to_fp16(v));
    }
  }
};

template <>
struct TensorToFP16<float> {
  void operator()(const Tensor& data, Tensor& dst) const {
    const auto count = narrow<size_t>(data.Shape().Size());
    MlasConvertFloatToHalfBuffer(data.Data<float>(), dst.MutableData<MLFloat16>(), count);
  }
};

template <typename T>
struct ToBFloat16;

template <>
struct ToBFloat16<BFloat16> {
  uint16_t operator()(const BFloat16& bf) const {
    return bf.val;
  }
};

template <>
struct ToBFloat16<float> {
  uint16_t operator()(float f) const {
    return BFloat16(f).val;
  }
};

template <>
struct ToBFloat16<double> {
  uint16_t operator()(double d) const {
    // The same code as in Eigen. We assume the loss of precision will occur
    // hence static_cast
    return BFloat16(static_cast<float>(d)).val;
  }
};

template <typename T>
struct TensorToProtoBFloat16 {
  void operator()(const Tensor& data, ONNX_NAMESPACE::TensorProto& proto) const {
    ToBFloat16<T> to_bfloat16;
    auto span = data.DataAsSpan<T>();
    for (const auto& v : span) {
      proto.add_int32_data(to_bfloat16(v));
    }
  }
};

template <typename T>
struct TensorToBFloat16 {
  void operator()(const Tensor& data, Tensor& dst) const {
    ToBFloat16<T> to_bfloat16;
    auto span = data.DataAsSpan<T>();
    auto* dst_data = dst.MutableData<BFloat16>();
    for (const auto& v : span) {
      *dst_data++ = BFloat16::FromBits(to_bfloat16(v));
    }
  }
};

template <typename T>
struct ToFloat32;

template <>
struct ToFloat32<float> {
  float operator()(const float& f) const {
    return f;
  }
};

template <>
struct ToFloat32<double> {
  float operator()(double d) const {
    return static_cast<float>(d);
  }
};

template <>
struct ToFloat32<BFloat16> {
  float operator()(BFloat16 bf) const {
    return static_cast<float>(bf);
  }
};

template <>
struct ToFloat32<MLFloat16> {
  float operator()(MLFloat16 hf) const {
    return static_cast<float>(hf);
  }
};

template <typename T>
struct TensorToFloat32 {
  void operator()(const Tensor& src, Tensor& dst, onnxruntime::concurrency::ThreadPool* /*thread_pool*/) const {
    auto src_span = src.DataAsSpan<T>();
    auto* dst_data = dst.MutableData<float>();
    ToFloat32<T> to_float32;
    for (const auto& v : src_span) {
      *dst_data++ = to_float32(v);
    }
  }
};

template <>
struct TensorToFloat32<MLFloat16> {
  void operator()(const Tensor& data,
                  Tensor& dst,
                  onnxruntime::concurrency::ThreadPool* thread_pool) const {
    const auto count = narrow<size_t>(data.Shape().Size());
    MlasConvertHalfToFloatBufferInParallel(data.Data<MLFloat16>(), dst.MutableData<float>(), count, thread_pool);
  }
};

inline void SetNameDims(const std::string& name,
                        gsl::span<const int64_t> dims,
                        ONNX_NAMESPACE::TensorProto_DataType dt,
                        ONNX_NAMESPACE::TensorProto& tensor_proto) {
  tensor_proto.set_name(name);
  tensor_proto.set_data_type(dt);

  for (auto d : dims) {
    tensor_proto.add_dims(d);
  }
}

}  // namespace

ONNX_NAMESPACE::TensorProto Initializer::ToFP16(const std::string& name) const {
  ONNX_NAMESPACE::TensorProto tensor_proto;
  SetNameDims(name, data_->Shape().GetDims(), ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, tensor_proto);
  utils::MLTypeCallDispatcher<MLFloat16, float, double> t_disp(data_->GetElementType());
  t_disp.Invoke<TensorToProtoFP16>(*data_, tensor_proto);
  return tensor_proto;
}

ONNX_NAMESPACE::TensorProto Initializer::ToBFloat16(const std::string& name) const {
  ONNX_NAMESPACE::TensorProto tensor_proto;
  SetNameDims(name, data_->Shape().GetDims(), ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16, tensor_proto);
  utils::MLTypeCallDispatcher<BFloat16, float, double> t_disp(data_->GetElementType());
  t_disp.Invoke<TensorToProtoBFloat16>(*data_, tensor_proto);
  return tensor_proto;
}

OrtValue onnxruntime::Initializer::ToFP16() const {
  if (data_->IsDataType<MLFloat16>()) {
    return ort_value_;
  }
  OrtValue result;
  auto tensor = Tensor(DataTypeImpl::GetType<MLFloat16>(), data_->Shape().GetDims(), CPUAllocator::DefaultInstance());
  Tensor::InitOrtValue(std::move(tensor), result);
  utils::MLTypeCallDispatcher<float, double> t_disp(data_->GetElementType());
  t_disp.Invoke<TensorToFP16>(*data_, *result.GetMutable<Tensor>());
  return result;
}

OrtValue Initializer::ToBFloat16() const {
  if (data_->IsDataType<BFloat16>()) {
    return ort_value_;
  }
  OrtValue result;
  auto tensor = Tensor(DataTypeImpl::GetType<BFloat16>(), data_->Shape().GetDims(), CPUAllocator::DefaultInstance());
  Tensor::InitOrtValue(std::move(tensor), result);
  utils::MLTypeCallDispatcher<float, double> t_disp(data_->GetElementType());
  t_disp.Invoke<TensorToBFloat16>(*data_, *result.GetMutable<Tensor>());
  return result;
}

OrtValue Initializer::ToFloat32(onnxruntime::concurrency::ThreadPool* thread_pool) const {
  if (data_->IsDataType<float>()) {
    return ort_value_;
  }
  OrtValue result;
  auto tensor = Tensor(DataTypeImpl::GetType<float>(), data_->Shape().GetDims(), CPUAllocator::DefaultInstance());
  Tensor::InitOrtValue(std::move(tensor), result);
  utils::MLTypeCallDispatcher<double, BFloat16, MLFloat16> t_disp(data_->GetElementType());
  t_disp.Invoke<TensorToFloat32>(*data_, *result.GetMutable<Tensor>(), thread_pool);
  return result;
}

namespace {

// std::identity c++20
template <typename T>
struct ToNumeric {
  using type = T;
  constexpr const T& operator()(const T& v) const {
    return v;
  }
};

template <>
struct ToNumeric<MLFloat16> {
  using type = float;
  float operator()(const MLFloat16& v) const {
    return v.ToFloat();
  }
};

template <>
struct ToNumeric<BFloat16> {
  using type = float;
  float operator()(const BFloat16& v) const {
    return v.ToFloat();
  }
};

template <typename T, typename Op>
struct OpElementWise {
  void Invoke(Tensor& lhs, const Tensor& rhs) const {
    Op op;
    ToNumeric<T> to_numeric;
    auto dst_span = lhs.MutableDataAsSpan<T>();
    auto src_span = rhs.DataAsSpan<T>();
    for (size_t i = 0, limit = dst_span.size(); i < limit; ++i) {
      dst_span[i] = T(op(to_numeric(dst_span[i]), to_numeric(src_span[i])));
    }
  }
};

template <typename T>
struct ScalarAdd {
  void operator()(Tensor& tensor, float v) const {
    ToNumeric<T> to_numeric;
    auto span = tensor.MutableDataAsSpan<T>();
    for (auto& dst : span) {
      dst = T(to_numeric(dst) + v);
    }
  }
};

template <typename T>
struct Sqrt {
  void operator()(Tensor& tensor) const {
    ToNumeric<T> to_numeric;
    auto span = tensor.MutableDataAsSpan<T>();
    for (auto& dst : span) {
      auto v = to_numeric(dst);
      dst = T(std::sqrt(v));
    }
  }
};

template <typename T>
struct ElementWiseAdd : OpElementWise<T, std::plus<typename ToNumeric<T>::type>> {
  void operator()(Tensor& lhs, const Tensor& rhs) const {
    this->Invoke(lhs, rhs);
  }
};

template <typename T>
struct ElementWiseSub : OpElementWise<T, std::minus<typename ToNumeric<T>::type>> {
  void operator()(Tensor& lhs, const Tensor& rhs) const {
    this->Invoke(lhs, rhs);
  }
};

template <typename T>
struct ElementWiseMul : OpElementWise<T, std::multiplies<typename ToNumeric<T>::type>> {
  void operator()(Tensor& lhs, const Tensor& rhs) const {
    this->Invoke(lhs, rhs);
  }
};

template <typename T>
struct ElementWiseDiv : OpElementWise<T, std::divides<typename ToNumeric<T>::type>> {
  void operator()(Tensor& lhs, const Tensor& rhs) const {
    this->Invoke(lhs, rhs);
  }
};
}  // namespace

Initializer& Initializer::add(float value) {
  utils::MLTypeCallDispatcher<MLFloat16, BFloat16, float, double> t_disp(data_->GetElementType());
  t_disp.Invoke<ScalarAdd>(*data_, value);
  return *this;
}

Initializer& Initializer::add(const Initializer& other) {
  ORT_ENFORCE(data_type() == other.data_type(), "Expecting the same data type");
  ORT_ENFORCE(size() == other.size(), "Expecting the same size");
  utils::MLTypeCallDispatcher<MLFloat16, BFloat16, float, double, int32_t, int64_t> t_disp(data_->GetElementType());
  t_disp.Invoke<ElementWiseAdd>(*data_, *other.data_);
  return *this;
}

Initializer& Initializer::sub(const Initializer& other) {
  ORT_ENFORCE(data_type() == other.data_type(), "Expecting the same data type");
  ORT_ENFORCE(size() == other.size(), "Expecting the same size");
  utils::MLTypeCallDispatcher<MLFloat16, BFloat16, float, double, int32_t, int64_t> t_disp(data_->GetElementType());
  t_disp.Invoke<ElementWiseSub>(*data_, *other.data_);
  return *this;
}

Initializer& Initializer::mul(const Initializer& other) {
  ORT_ENFORCE(data_type() == other.data_type(), "Expecting the same data type");
  ORT_ENFORCE(size() == other.size(), "Expecting the same size");
  utils::MLTypeCallDispatcher<MLFloat16, BFloat16, float, double, int32_t, int64_t> t_disp(data_->GetElementType());
  t_disp.Invoke<ElementWiseMul>(*data_, *other.data_);
  return *this;
}

Initializer& Initializer::div(const Initializer& other) {
  ORT_ENFORCE(data_type() == other.data_type(), "Expecting the same data type");
  ORT_ENFORCE(size() == other.size(), "Expecting the same size");
  utils::MLTypeCallDispatcher<MLFloat16, BFloat16, float, double, int32_t, int64_t> t_disp(data_->GetElementType());
  t_disp.Invoke<ElementWiseDiv>(*data_, *other.data_);
  return *this;
}

Initializer& Initializer::sqrt() {
  utils::MLTypeCallDispatcher<MLFloat16, BFloat16, float, double> t_disp(data_->GetElementType());
  t_disp.Invoke<Sqrt>(*data_);
  return *this;
}

namespace {
template <typename T>
struct ScaleByAxis {
  void operator()(Tensor& data,
                  const Tensor& scalers,
                  const size_t block_size,
                  const size_t num_blocks,
                  const bool column_major) const {
    ToNumeric<T> to_numeric;
    const auto scaler_size = scalers.Shape().Size();
    T* dst = data.MutableData<T>();
    const T* scalers_data = scalers.Data<T>();
    if (scaler_size == 1) {
      const auto numeric_scaler = to_numeric(scalers_data[0]);
      for (size_t block_offset = 0, limit = block_size * num_blocks; block_offset < limit; ++block_offset) {
        dst[block_offset] = T(to_numeric(dst[block_offset]) * numeric_scaler);
      }
    } else {
      for (size_t block_offset = 0, i = 0; i < num_blocks; i++) {
        if (column_major) {
          for (size_t j = 0; j < block_size; ++j, ++block_offset) {
            const auto numeric_scaler = to_numeric(scalers_data[j]);
            dst[block_offset] = T(to_numeric(dst[block_offset]) * numeric_scaler);
          }
        } else {
          const auto numeric_scaler = to_numeric(scalers_data[i]);
          for (size_t j = 0; j < block_size; ++j, ++block_offset) {
            dst[block_offset] = T(to_numeric(dst[block_offset]) * numeric_scaler);
          }
        }
      }
    }
  }
};
}  // namespace

void Initializer::scale_by_axis(const Initializer& scalers, int axis, bool column_major) {
  ORT_ENFORCE(axis >= 0, "Axis must be non-negative");
  const size_t block_size = narrow<size_t>(data_->Shape().SizeFromDimension(gsl::narrow_cast<size_t>(axis)));
  const size_t num_blocks = size() / block_size;
  ORT_ENFORCE(scalers.size() == 1 ||
                  (column_major ? scalers.size() == block_size : scalers.size() == num_blocks),
              "Invalid other(scalers) size");
  utils::MLTypeCallDispatcher<MLFloat16, BFloat16, float, double, int32_t, int64_t> t_disp(data_->GetElementType());
  t_disp.Invoke<ScaleByAxis>(*data_, *scalers.data_, block_size, num_blocks, column_major);
}
#endif  // ORT_EXTENDED_MINIMAL_BUILD
}  // namespace onnxruntime
