// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"

#include <functional>
#include <memory>

#include "core/common/gsl.h"
#include "core/common/path.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/tensor_external_data_info.h"
#include "core/platform/env.h"

namespace onnxruntime {

Initializer::Initializer(ONNX_NAMESPACE::TensorProto_DataType data_type,
                         std::string_view name,
                         gsl::span<const int64_t> dims)
    : name_(name),
      data_(DataTypeImpl::TensorTypeFromONNXEnum(data_type)->GetElementType(), dims,
            std::make_shared<CPUAllocator>()) {
  if (!data_.IsDataTypeString()) {
    memset(data_.MutableDataRaw(), 0, data_.SizeInBytes());
  }
}

Initializer::Initializer(const ONNX_NAMESPACE::TensorProto& tensor_proto, const Path& model_path) {
  ORT_ENFORCE(utils::HasDataType(tensor_proto), "Initializer must have a datatype");
  if (utils::HasExternalData(tensor_proto)) {
    ORT_ENFORCE(!model_path.IsEmpty(),
                "model_path must not be empty. Ensure that a path is provided when the model is created or loaded.");
  }

  auto proto_data_type = tensor_proto.data_type();
  if (utils::HasName(tensor_proto)) {
    name_ = tensor_proto.name();
  }

  auto proto_shape = utils::GetTensorShapeFromTensorProto(tensor_proto);

  // This must be pre-allocated
  Tensor w(DataTypeImpl::TensorTypeFromONNXEnum(proto_data_type)->GetElementType(), proto_shape,
           std::make_shared<CPUAllocator>());
  ORT_THROW_IF_ERROR(utils::TensorProtoToTensor(Env::Default(), model_path.ToPathString().c_str(), tensor_proto, w));
  data_ = std::move(w);
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
  SetNameDims(name, data_.Shape().GetDims(), ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, tensor_proto);
  utils::MLTypeCallDispatcher<MLFloat16, float, double> t_disp(data_.GetElementType());
  t_disp.Invoke<TensorToProtoFP16>(data_, tensor_proto);
  return tensor_proto;
}

ONNX_NAMESPACE::TensorProto Initializer::ToBFloat16(const std::string& name) const {
  ONNX_NAMESPACE::TensorProto tensor_proto;
  SetNameDims(name, data_.Shape().GetDims(), ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16, tensor_proto);
  utils::MLTypeCallDispatcher<BFloat16, float, double> t_disp(data_.GetElementType());
  t_disp.Invoke<TensorToProtoBFloat16>(data_, tensor_proto);
  return tensor_proto;
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
  utils::MLTypeCallDispatcher<MLFloat16, BFloat16, float, double> t_disp(data_.GetElementType());
  t_disp.Invoke<ScalarAdd>(data_, value);
  return *this;
}

Initializer& Initializer::add(const Initializer& other) {
  ORT_ENFORCE(data_type() == other.data_type(), "Expecting the same data type");
  ORT_ENFORCE(size() == other.size(), "Expecting the same size");
  utils::MLTypeCallDispatcher<MLFloat16, BFloat16, float, double, int32_t, int64_t> t_disp(data_.GetElementType());
  t_disp.Invoke<ElementWiseAdd>(data_, other.data_);
  return *this;
}

Initializer& Initializer::sub(const Initializer& other) {
  ORT_ENFORCE(data_type() == other.data_type(), "Expecting the same data type");
  ORT_ENFORCE(size() == other.size(), "Expecting the same size");
  utils::MLTypeCallDispatcher<MLFloat16, BFloat16, float, double, int32_t, int64_t> t_disp(data_.GetElementType());
  t_disp.Invoke<ElementWiseSub>(data_, other.data_);
  return *this;
}

Initializer& Initializer::mul(const Initializer& other) {
  ORT_ENFORCE(data_type() == other.data_type(), "Expecting the same data type");
  ORT_ENFORCE(size() == other.size(), "Expecting the same size");
  utils::MLTypeCallDispatcher<MLFloat16, BFloat16, float, double, int32_t, int64_t> t_disp(data_.GetElementType());
  t_disp.Invoke<ElementWiseMul>(data_, other.data_);
  return *this;
}

Initializer& Initializer::div(const Initializer& other) {
  ORT_ENFORCE(data_type() == other.data_type(), "Expecting the same data type");
  ORT_ENFORCE(size() == other.size(), "Expecting the same size");
  utils::MLTypeCallDispatcher<MLFloat16, BFloat16, float, double, int32_t, int64_t> t_disp(data_.GetElementType());
  t_disp.Invoke<ElementWiseDiv>(data_, other.data_);
  return *this;
}

Initializer& Initializer::sqrt() {
  utils::MLTypeCallDispatcher<MLFloat16, BFloat16, float, double> t_disp(data_.GetElementType());
  t_disp.Invoke<Sqrt>(data_);
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
  const size_t block_size = narrow<size_t>(data_.Shape().SizeFromDimension(gsl::narrow_cast<size_t>(axis)));
  const size_t num_blocks = size() / block_size;
  ORT_ENFORCE(scalers.size() == 1 ||
                  (column_major ? scalers.size() == block_size : scalers.size() == num_blocks),
              "Invalid other(scalers) size");
  utils::MLTypeCallDispatcher<MLFloat16, BFloat16, float, double, int32_t, int64_t> t_disp(data_.GetElementType());
  t_disp.Invoke<ScaleByAxis>(data_, scalers.data_, block_size, num_blocks, column_major);
}
#endif  // ORT_EXTENDED_MINIMAL_BUILD
}  // namespace onnxruntime
