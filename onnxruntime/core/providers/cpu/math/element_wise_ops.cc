// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/math/element_wise_ops.h"
#include <unsupported/Eigen/SpecialFunctions>
#include "core/mlas/inc/mlas.h"
#include <x86intrin.h>

namespace onnxruntime {

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Add,
    7,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Add<float>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Add,
    7,
    int32_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()),
    Add<int32_t>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Add,
    7,
    int64_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>()),
    Add<int64_t>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Sub,
    7,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Sub<float>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Sub,
    7,
    int32_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()),
    Sub<int32_t>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Sub,
    7,
    int64_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>()),
    Sub<int64_t>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Mul,
    7,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Mul<float>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Mul,
    7,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    Mul<double>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Mul,
    7,
    int32_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()),
    Mul<int32_t>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Mul,
    7,
    int64_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>()),
    Mul<int64_t>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Div,
    7,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Div<float>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Div,
    7,
    int32_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()),
    Div<int32_t>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Div,
    7,
    int64_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>()),
    Div<int64_t>);

#define REG_ABS_KERNEL(TYPE)                                                       \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                                  \
      Abs,                                                                         \
      6,                                                                           \
      TYPE,                                                                        \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<TYPE>()), \
      Abs<TYPE>);

REG_ABS_KERNEL(float)
REG_ABS_KERNEL(double)
REG_ABS_KERNEL(int8_t)
REG_ABS_KERNEL(int16_t)
REG_ABS_KERNEL(int32_t)
REG_ABS_KERNEL(int64_t)
REG_ABS_KERNEL(uint8_t)
REG_ABS_KERNEL(uint16_t)
REG_ABS_KERNEL(uint32_t)
REG_ABS_KERNEL(uint64_t)

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Neg,
    6,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Neg<float>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Neg,
    6,
    int8_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int8_t>()),
    Neg<int8_t>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Neg,
    6,
    int32_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()),
    Neg<int32_t>);

ONNX_CPU_OPERATOR_KERNEL(
    Floor,
    6,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Floor<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Ceil,
    6,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Ceil<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Reciprocal,
    6,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Reciprocal<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Sqrt,
    6,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Sqrt<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Pow,
    7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Pow<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Exp,
    6,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Exp<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Log,
    6,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Log<float>);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Sum,
    6, 7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Sum_6<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Sum,
    8,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Sum_8<float>);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Min,
    6, 7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Min_6<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Min,
    8,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Min_8<float>);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Max,
    6, 7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Max_6<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Max,
    8,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Max_8<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Not,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<bool>()),
    Not);

ONNX_CPU_OPERATOR_KERNEL(
    And,
    7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<bool>()),
    And);

ONNX_CPU_OPERATOR_KERNEL(
    Or,
    7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<bool>()),
    Or);

ONNX_CPU_OPERATOR_KERNEL(
    Xor,
    7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<bool>()),
    Xor);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    Less,
    7, 9,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Less<float>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Less,
    9,
    int32_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()),
    Less<int32_t>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    Greater,
    7, 9,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Greater<float>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Greater,
    9,
    int32_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()),
    Greater<int32_t>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Equal,
    7,
    bool,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<bool>()),
    Equal<bool>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Equal,
    7,
    int32_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()),
    Equal<int32_t>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Equal,
    7,
    int64_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>()),
    Equal<int64_t>);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Mean,
    6, 7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Mean_6<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Mean,
    8,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Mean_8<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Affine,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Affine<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Scale,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Scale<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Erf,
    9,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Erf<float>);

template <typename T>
Status Add<T>::Compute(OpKernelContext* context) const {
  return BroadcastTwo<T, T>(
      *context,
      [](EigenVectorMap<T> output, T input0, ConstEigenVectorMap<T> input1) { output = input0 + input1.array(); },
      [](EigenVectorMap<T> output, ConstEigenVectorMap<T> input0, T input1) { output = input0.array() + input1; },
      [](EigenVectorMap<T> output, ConstEigenVectorMap<T> input0, ConstEigenVectorMap<T> input1) { output = input0 + input1; });
}

template <typename T>
Status Sub<T>::Compute(OpKernelContext* context) const {
  return BroadcastTwo<T, T>(
      *context,
      [](EigenVectorMap<T> output, T input0, ConstEigenVectorMap<T> input1) { output = input0 - input1.array(); },
      [](EigenVectorMap<T> output, ConstEigenVectorMap<T> input0, T input1) { output = input0.array() - input1; },
      [](EigenVectorMap<T> output, ConstEigenVectorMap<T> input0, ConstEigenVectorMap<T> input1) { output = input0 - input1; });
}

template <typename T>
Status Mul<T>::Compute(OpKernelContext* context) const {
  return BroadcastTwo<T, T>(
      *context,
      [](EigenVectorMap<T> output, T input0, ConstEigenVectorMap<T> input1) { output = input0 * input1.array(); },
      [](EigenVectorMap<T> output, ConstEigenVectorMap<T> input0, T input1) { output = input0.array() * input1; },
      [](EigenVectorMap<T> output, ConstEigenVectorMap<T> input0, ConstEigenVectorMap<T> input1) { output = input0.cwiseProduct(input1); });
}

template <typename T>
Status Div<T>::Compute(OpKernelContext* context) const {
  return BroadcastTwo<T, T>(
      *context,
      [](EigenVectorMap<T> output, T input0, ConstEigenVectorMap<T> input1) { output = input0 / input1.array(); },
      [](EigenVectorMap<T> output, ConstEigenVectorMap<T> input0, T input1) { output = input0.array() / input1; },
      [](EigenVectorMap<T> output, ConstEigenVectorMap<T> input0, ConstEigenVectorMap<T> input1) { output = input0.cwiseQuotient(input1); });
}

template <>
Status Floor<float>::Compute(OpKernelContext* ctx) const {
  auto& X = *ctx->Input<Tensor>(0);
  auto& Y = *ctx->Output(0, X.Shape());

  EigenMap<float>(Y) = EigenMap<float>(X).array().floor();

  return Status::OK();
}

template <>
Status Ceil<float>::Compute(OpKernelContext* ctx) const {
  auto& X = *ctx->Input<Tensor>(0);
  auto& Y = *ctx->Output(0, X.Shape());

  EigenMap<float>(Y) = EigenMap<float>(X).array().ceil();

  return Status::OK();
}

template <>
Status Reciprocal<float>::Compute(OpKernelContext* ctx) const {
  auto& X = *ctx->Input<Tensor>(0);
  auto& Y = *ctx->Output(0, X.Shape());

  EigenMap<float>(Y) = EigenMap<float>(X).cwiseInverse();

  return Status::OK();
}

template <>
Status Sqrt<float>::Compute(OpKernelContext* ctx) const {
  auto& X = *ctx->Input<Tensor>(0);
  auto& Y = *ctx->Output(0, X.Shape());

  EigenMap<float>(Y) = EigenMap<float>(X).cwiseSqrt();

  return Status::OK();
}

//template <>
//Status Pow<float>::Compute(OpKernelContext* context) const {
//  const Tensor& Y = *context->Input<Tensor>(1);
//  std::function<void(EigenVectorMap<float>, ConstEigenVectorMap<float>, float)> input1scalar =
//      [](EigenVectorMap<float> output, ConstEigenVectorMap<float> input0, float input1) { output = Eigen::pow(input0.array(), input1); };
//  if (Y.Shape().Size() == 1) {
//    float value = *Y.Data<float>();
//    if (value == 2.0) {
//      input1scalar = [](EigenVectorMap<float> output, ConstEigenVectorMap<float> input0, float) { output = Eigen::square(input0.array()); };
//    } else if (value == 3.0) {
//      input1scalar = [](EigenVectorMap<float> output, ConstEigenVectorMap<float> input0, float) { output = Eigen::cube(input0.array()); };
//    }
//  }
//
//  return BroadcastTwo<float, float>(
//      *context,
//      [](EigenVectorMap<float> output, float input0, ConstEigenVectorMap<float> input1) { output = Eigen::pow(input0, input1.array()); },
//      input1scalar,
//      [](EigenVectorMap<float> output, ConstEigenVectorMap<float> input0, ConstEigenVectorMap<float> input1) { output = Eigen::pow(input0.array(), input1.array()); });
//}

template <>
Status Exp<float>::Compute(OpKernelContext* ctx) const {
  auto& X = *ctx->Input<Tensor>(0);
  auto& Y = *ctx->Output(0, X.Shape());

  EigenMap<float>(Y) = EigenMap<float>(X).array().exp();

  return Status::OK();
}

template <>
Status Log<float>::Compute(OpKernelContext* ctx) const {
  auto& X = *ctx->Input<Tensor>(0);
  auto& Y = *ctx->Output(0, X.Shape());

  EigenMap<float>(Y) = EigenMap<float>(X).array().log();

  return Status::OK();
}

template <>
Status Sum_6<float>::Compute(OpKernelContext* ctx) const {
  auto input_count = Node().InputArgCount().front();
  ORT_ENFORCE(input_count >= 1, "Must have 1 or more inputs");
  auto& data_0 = *ctx->Input<Tensor>(0);
  auto& shape = data_0.Shape();
  auto sum = EigenMap<float>(*ctx->Output(0, shape));

  if (input_count == 1) {
    sum = EigenMap<float>(data_0);
  } else {
    auto& data_1 = *ctx->Input<Tensor>(1);
    ORT_ENFORCE(data_1.Shape() == shape, "All inputs must have the same shape");

    sum = EigenMap<float>(data_0) + EigenMap<float>(data_1);
    for (int index = 2; index < input_count; index++) {
      auto& data_n = *ctx->Input<Tensor>(index);
      ORT_ENFORCE(data_n.Shape() == shape, "All inputs must have the same shape");
      sum += EigenMap<float>(data_n);
    }
  }

  return Status::OK();
}

template <>
Status Sum_8<float>::Compute(OpKernelContext* context) const {
  return BroadcastVariadic<float, float>(
      Node(), *context,
      [](EigenVectorMap<float> output, float input0, ConstEigenVectorMap<float> input1) { output = input0 + input1.array(); },
      [](EigenVectorMap<float> output, ConstEigenVectorMap<float> input0, float input1) { output = input0.array() + input1; },
      [](EigenVectorMap<float> output, ConstEigenVectorMap<float> input0, ConstEigenVectorMap<float> input1) { output = input0 + input1; });
}

template <>
Status Min_6<float>::Compute(OpKernelContext* ctx) const {
  auto inputCount = Node().InputArgCount().front();
  ORT_ENFORCE(inputCount >= 1, "Must have 1 or more inputs");
  auto& data_0 = *ctx->Input<Tensor>(0);
  auto& shape = data_0.Shape();
  auto min = EigenMap<float>(*ctx->Output(0, shape));

  min = EigenMap<float>(data_0);
  for (int index = 1; index < inputCount; index++) {
    auto& data_n = *ctx->Input<Tensor>(index);
    ORT_ENFORCE(data_n.Shape() == shape, "All inputs must have the same shape");
    min = min.array().min(EigenMap<float>(data_n).array());
  }

  return Status::OK();
}

template <>
Status Min_8<float>::Compute(OpKernelContext* context) const {
  return BroadcastVariadic<float, float>(
      Node(), *context,
      [](EigenVectorMap<float> output, float input0, ConstEigenVectorMap<float> input1) { output = input1.array().min(input0); },
      [](EigenVectorMap<float> output, ConstEigenVectorMap<float> input0, float input1) { output = input0.array().min(input1); },
      [](EigenVectorMap<float> output, ConstEigenVectorMap<float> input0, ConstEigenVectorMap<float> input1) { output = input0.array().min(input1.array()); });
}

template <>
Status Max_6<float>::Compute(OpKernelContext* ctx) const {
  auto inputCount = Node().InputArgCount().front();
  ORT_ENFORCE(inputCount >= 1, "Must have 1 or more inputs");
  auto& data_0 = *ctx->Input<Tensor>(0);
  auto& shape = data_0.Shape();
  auto max = EigenMap<float>(*ctx->Output(0, shape));

  max = EigenMap<float>(data_0);
  for (int index = 1; index < inputCount; index++) {
    auto& data_n = *ctx->Input<Tensor>(index);
    ORT_ENFORCE(data_n.Shape() == shape, "All inputs must have the same shape");
    max = max.array().max(EigenMap<float>(data_n).array());
  }

  return Status::OK();
}

template <>
Status Max_8<float>::Compute(OpKernelContext* context) const {
  return BroadcastVariadic<float, float>(
      Node(), *context,
      [](EigenVectorMap<float> output, float input0, ConstEigenVectorMap<float> input1) { output = input1.array().max(input0); },
      [](EigenVectorMap<float> output, ConstEigenVectorMap<float> input0, float input1) { output = input0.array().max(input1); },
      [](EigenVectorMap<float> output, ConstEigenVectorMap<float> input0, ConstEigenVectorMap<float> input1) { output = input0.array().max(input1.array()); });
}

Status Not::Compute(OpKernelContext* context) const {
  auto& input = *context->Input<Tensor>(0);
  auto& output = *context->Output(0, input.Shape());

  EigenMap<bool>(output).array() = !EigenMap<bool>(input).array();
  return Status::OK();
}

Status And::Compute(OpKernelContext* context) const {
  // The scalar cases are special cased, since 'X && true = X' and 'X && false = false'
  return BroadcastTwo<bool, bool>(
      *context,
      [](EigenVectorMap<bool> output, bool input0, ConstEigenVectorMap<bool> input1) {
        if (input0)
          output = input1;
        else
          output.array() = false;
      },
      [](EigenVectorMap<bool> output, ConstEigenVectorMap<bool> input0, bool input1) {
        if (input1)
          output = input0;
        else
          output.array() = false;
      },
      [](EigenVectorMap<bool> output, ConstEigenVectorMap<bool> input0, ConstEigenVectorMap<bool> input1) { output = input0.array() && input1.array(); });
}

Status Or::Compute(OpKernelContext* context) const {
  // The scalar cases are special cased, since 'X || true = true' and 'X || false = X'
  return BroadcastTwo<bool, bool>(
      *context,
      [](EigenVectorMap<bool> output, bool input0, ConstEigenVectorMap<bool> input1) {
        if (input0)
          output.array() = true;
        else
          output = input1;
      },
      [](EigenVectorMap<bool> output, ConstEigenVectorMap<bool> input0, bool input1) {
        if (input1)
          output.array() = true;
        else
          output = input0;
      },
      [](EigenVectorMap<bool> output, ConstEigenVectorMap<bool> input0, ConstEigenVectorMap<bool> input1) { output = input0.array() || input1.array(); });
}

Status Xor::Compute(OpKernelContext* context) const {
  // The scalar cases are special cased, since 'X ^ true = !X' and 'X ^ false = X'
  return BroadcastTwo<bool, bool>(
      *context,
      [](EigenVectorMap<bool> output, bool input0, ConstEigenVectorMap<bool> input1) {
        if (input0)
          output.array() = !input1.array();
        else
          output = input1;
      },
      [](EigenVectorMap<bool> output, ConstEigenVectorMap<bool> input0, bool input1) {
        if (input1)
          output.array() = !input0.array();
        else
          output = input0;
      },
      [](EigenVectorMap<bool> output, ConstEigenVectorMap<bool> input0, ConstEigenVectorMap<bool> input1) { output = input0.array() ^ input1.array(); });
}

template <typename T>
Status Equal<T>::Compute(OpKernelContext* context) const {
  return BroadcastTwo<T, bool>(
      *context,
      [](EigenVectorMap<bool> output, T input0, ConstEigenVectorMap<T> input1) { output = input1.array() == input0; },
      [](EigenVectorMap<bool> output, ConstEigenVectorMap<T> input0, T input1) { output = input0.array() == input1; },
      [](EigenVectorMap<bool> output, ConstEigenVectorMap<T> input0, ConstEigenVectorMap<T> input1) { output = input0.array() == input1.array(); });
}

template <typename T>
Status Less<T>::Compute(OpKernelContext* context) const {
  return BroadcastTwo<T, bool>(
      *context,
      [](EigenVectorMap<bool> output, T input0, ConstEigenVectorMap<T> input1) { output = input1.array() > input0; },
      [](EigenVectorMap<bool> output, ConstEigenVectorMap<T> input0, T input1) { output = input0.array() < input1; },
      [](EigenVectorMap<bool> output, ConstEigenVectorMap<T> input0, ConstEigenVectorMap<T> input1) { output = input0.array() < input1.array(); });
}

template <typename T>
Status Greater<T>::Compute(OpKernelContext* context) const {
  return BroadcastTwo<T, bool>(
      *context,
      [](EigenVectorMap<bool> output, T input0, ConstEigenVectorMap<T> input1) { output = input1.array() < input0; },
      [](EigenVectorMap<bool> output, ConstEigenVectorMap<T> input0, T input1) { output = input0.array() > input1; },
      [](EigenVectorMap<bool> output, ConstEigenVectorMap<T> input0, ConstEigenVectorMap<T> input1) { output = input0.array() > input1.array(); });
}

template <>
Status Mean_6<float>::Compute(OpKernelContext* ctx) const {
  auto inputCount = Node().InputArgCount().front();
  ORT_ENFORCE(inputCount >= 1, "Must have 1 or more inputs");
  auto& data_0 = *ctx->Input<Tensor>(0);
  auto& shape = data_0.Shape();
  auto mean = EigenMap<float>(*ctx->Output(0, shape));

  if (inputCount == 1) {
    mean = EigenMap<float>(data_0);
  } else {
    auto& data_1 = *ctx->Input<Tensor>(1);
    ORT_ENFORCE(data_1.Shape() == shape, "All inputs must have the same shape");

    mean = EigenMap<float>(data_0) + EigenMap<float>(data_1);
    for (int index = 2; index < inputCount; index++) {
      auto& data_n = *ctx->Input<Tensor>(index);
      ORT_ENFORCE(data_n.Shape() == shape, "All inputs must have the same shape");
      mean += EigenMap<float>(data_n);
    }
  }

  // Take the mean
  float weight = 1.0f / static_cast<float>(inputCount);
  mean = mean * weight;

  return Status::OK();
}

template <>
Status Mean_8<float>::Compute(OpKernelContext* context) const {
  // Do a sum exactly the same as in Sum_8
  Status status = BroadcastVariadic<float, float>(
      Node(), *context,
      [](EigenVectorMap<float> output, float input0, ConstEigenVectorMap<float> input1) { output = input0 + input1.array(); },
      [](EigenVectorMap<float> output, ConstEigenVectorMap<float> input0, float input1) { output = input0.array() + input1; },
      [](EigenVectorMap<float> output, ConstEigenVectorMap<float> input0, ConstEigenVectorMap<float> input1) { output = input0 + input1; });
  if (!status.IsOK())
    return status;

  // Now divide by the input count to get the mean
  EigenMap<float>(*context->Output<Tensor>(0)) *= 1.0f / static_cast<float>(Node().InputArgCount().front());
  return Status::OK();
}

template <>
Status Affine<float>::Compute(OpKernelContext* ctx) const {
  auto& X = *ctx->Input<Tensor>(0);
  auto& Y = *ctx->Output(0, X.Shape());
  MakeEigenArrayMap<float>(Y) = alpha_ * MakeEigenArrayMap<float>(X) + beta_;
  return Status::OK();
}

template <typename T>
class Sin final : public OpKernel {
 public:
  Sin(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    auto& X = *context->Input<Tensor>(0);
    auto& Y = *context->Output(0, X.Shape());
    MakeEigenArrayMap<float>(Y) = MakeEigenArrayMap<float>(X).sin();
    return Status::OK();
  }
};

ONNX_CPU_OPERATOR_KERNEL(
    Sin,
    7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Sin<float>);

template <typename T>
class Cos final : public OpKernel {
 public:
  Cos(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    auto& X = *context->Input<Tensor>(0);
    auto& Y = *context->Output(0, X.Shape());
    MakeEigenArrayMap<float>(Y) = MakeEigenArrayMap<float>(X).cos();
    return Status::OK();
  }
};

ONNX_CPU_OPERATOR_KERNEL(
    Cos,
    7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Cos<float>);

template <typename T>
class Tan final : public OpKernel {
 public:
  Tan(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    auto& X = *context->Input<Tensor>(0);
    auto& Y = *context->Output(0, X.Shape());
    MakeEigenArrayMap<float>(Y) = MakeEigenArrayMap<float>(X).tan();
    return Status::OK();
  }
};

ONNX_CPU_OPERATOR_KERNEL(
    Tan,
    7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Tan<float>);

template <typename T>
class Asin final : public OpKernel {
 public:
  Asin(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    auto& X = *context->Input<Tensor>(0);
    auto& Y = *context->Output(0, X.Shape());
    MakeEigenArrayMap<float>(Y) = MakeEigenArrayMap<float>(X).asin();
    return Status::OK();
  }
};

ONNX_CPU_OPERATOR_KERNEL(
    Asin,
    7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Asin<float>);

template <typename T>
class Acos final : public OpKernel {
 public:
  Acos(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    auto& X = *context->Input<Tensor>(0);
    auto& Y = *context->Output(0, X.Shape());
    MakeEigenArrayMap<float>(Y) = MakeEigenArrayMap<float>(X).acos();
    return Status::OK();
  }
};

ONNX_CPU_OPERATOR_KERNEL(
    Acos,
    7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Acos<float>);

template <typename T>
class Atan final : public OpKernel {
 public:
  Atan(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    auto& X = *context->Input<Tensor>(0);
    auto& Y = *context->Output(0, X.Shape());
    MakeEigenArrayMap<float>(Y) = MakeEigenArrayMap<float>(X).atan();
    return Status::OK();
  }
};

ONNX_CPU_OPERATOR_KERNEL(
    Atan,
    7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Atan<float>);

template <typename T>
class Sinh final : public OpKernel {
 public:
  explicit Sinh(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    auto& X = *context->Input<Tensor>(0);
    auto& Y = *context->Output(0, X.Shape());
    MakeEigenArrayMap<float>(Y) = MakeEigenArrayMap<float>(X).sinh();
    return Status::OK();
  }
};

ONNX_CPU_OPERATOR_KERNEL(
    Sinh,
    9,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Sinh<float>);

template <typename T>
class Cosh final : public OpKernel {
 public:
  explicit Cosh(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    auto& X = *context->Input<Tensor>(0);
    auto& Y = *context->Output(0, X.Shape());
    MakeEigenArrayMap<float>(Y) = MakeEigenArrayMap<float>(X).cosh();
    return Status::OK();
  }
};

ONNX_CPU_OPERATOR_KERNEL(
    Cosh,
    9,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Cosh<float>);

template <typename T>
class Asinh final : public OpKernel {
 public:
  explicit Asinh(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    auto& X = *context->Input<Tensor>(0);
    auto& Y = *context->Output(0, X.Shape());

    auto X_data = X.template Data<float>();
    auto Y_data = Y.template MutableData<float>();

    auto in = gsl::make_span(X_data, X.Shape().Size());
    auto out = gsl::make_span(Y_data, Y.Shape().Size());

    for (int64_t index = 0; index < in.size(); ++index) {
      out[index] = std::asinh(in[index]);
    }
    return Status::OK();
  }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Asinh);
};

ONNX_CPU_OPERATOR_KERNEL(
    Asinh,
    9,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Asinh<float>);

template <typename T>
class Acosh final : public OpKernel {
 public:
  explicit Acosh(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    auto& X = *context->Input<Tensor>(0);
    auto& Y = *context->Output(0, X.Shape());

    auto X_data = X.template Data<float>();
    auto Y_data = Y.template MutableData<float>();

    auto in = gsl::make_span(X_data, X.Shape().Size());
    auto out = gsl::make_span(Y_data, Y.Shape().Size());

    for (int64_t index = 0; index < in.size(); ++index) {
      out[index] = std::acosh(in[index]);
    }
    return Status::OK();
  }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Acosh);
};

ONNX_CPU_OPERATOR_KERNEL(
    Acosh,
    9,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Acosh<float>);

template <typename T>
class Atanh final : public OpKernel {
 public:
  explicit Atanh(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    auto& X = *context->Input<Tensor>(0);
    auto& Y = *context->Output(0, X.Shape());

    auto X_data = X.template Data<float>();
    auto Y_data = Y.template MutableData<float>();

    auto in = gsl::make_span(X_data, X.Shape().Size());
    auto out = gsl::make_span(Y_data, Y.Shape().Size());

    for (int64_t index = 0; index < in.size(); ++index) {
      out[index] = std::atanh(in[index]);
    }
    return Status::OK();
  }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Atanh);
};

ONNX_CPU_OPERATOR_KERNEL(
    Atanh,
    9,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Atanh<float>);

template <>
Status PRelu<float>::Compute(OpKernelContext* context) const {
  return BroadcastTwo<float, float>(
      *context,
      [](EigenVectorMap<float> output, float input0, ConstEigenVectorMap<float> input1) {
        if (input0 > 0)
          output.array() = input0;
        else
          output = input0 * input1.array();
      },
      [](EigenVectorMap<float> output, ConstEigenVectorMap<float> input0, float input1) {
        output = (input0.array() > 0).select(input0, input0 * input1);
      },
      [](EigenVectorMap<float> output, ConstEigenVectorMap<float> input0, ConstEigenVectorMap<float> input1) {
        output = (input0.array() > 0).select(input0, input0.cwiseProduct(input1));
      });
}

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    PRelu,
    7,
    9,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    PRelu<float>);

// This is a special case version of TBroadcaster just for Expand that only has a shape as the second parameter
template <typename T>
struct TBroadcasterExpand {
  TBroadcasterExpand(const Tensor& input, const std::vector<int64_t>& shape)
      : input_tensor_(input),
        broadcaster_(input.Shape().GetDims(), shape) {
  }

  TensorShape GetOutputShape() const { return TensorShape(broadcaster_.output_shape_); }
  size_t GetSpanSize() const { return span_size_; }

  bool IsInput0Scalar() const { return broadcaster_.iterator1_.deltas_.front() == 0; }

  T NextScalar() { return *Next(); }

  ConstEigenVectorMap<T> NextEigen() { return ConstEigenVectorMap<T>(Next(), span_size_); }

 private:
  const T* Next() { return input_ + broadcaster_.iterator1_.AdvanceBy(span_size_); }

  const Tensor& input_tensor_;
  Broadcaster broadcaster_;
  size_t span_size_{broadcaster_.GetSpanSize()};

  const T* input_{input_tensor_.template Data<T>()};
};

template <typename T>
Status Expand_8<T>::Compute(OpKernelContext* context) const {
  auto& tensor_shape = *context->Input<Tensor>(1);
  ORT_ENFORCE(tensor_shape.Shape().GetDims().size() == 1, "Shape must be 1 dimensional as it's tensor data is a shape");

  // Turn the shape tensor data into an actual shape
  const int64_t* p_shape = tensor_shape.template Data<int64_t>();
  std::vector<int64_t> shape{p_shape, p_shape + tensor_shape.Shape().Size()};

  TBroadcasterExpand<T> bc(*context->Input<Tensor>(0), shape);
  TBroadcastOutput<T> output(bc.GetSpanSize(), *context->Output(0, bc.GetOutputShape()));

  // This doesn't use BroadcastLoop since there is no second tensor, just duplicating the first
  if (bc.IsInput0Scalar()) {
    // Input0 being a scalar is the only special case here, since we're duplicating a single value
    while (output)
      output.NextEigenOutput().array() = bc.NextScalar();
  } else {
    // Input1 being a scalar doesn't matter (as there's no actual input1). We're still duplicating Input0 in the same sized chunks
    while (output)
      output.NextEigenOutput() = bc.NextEigen();
  }
  return Status::OK();
}

#define REG_EXPAND_KERNEL(TYPE)                                                     \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                                   \
      Expand,                                                                       \
      8,                                                                            \
      TYPE,                                                                         \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<TYPE>()),  \
      Expand_8<TYPE>);

REG_EXPAND_KERNEL(float)
REG_EXPAND_KERNEL(double)
REG_EXPAND_KERNEL(int8_t)
REG_EXPAND_KERNEL(int16_t)
REG_EXPAND_KERNEL(int32_t)
REG_EXPAND_KERNEL(int64_t)
REG_EXPAND_KERNEL(uint8_t)
REG_EXPAND_KERNEL(uint16_t)
REG_EXPAND_KERNEL(uint32_t)
REG_EXPAND_KERNEL(uint64_t)
REG_EXPAND_KERNEL(bool)
REG_EXPAND_KERNEL(MLFloat16)

template <>
Status Scale<float>::Compute(OpKernelContext* ctx) const {
  auto& X = *ctx->Input<Tensor>(0);
  auto& Y = *ctx->Output(0, X.Shape());
  EigenMap<float>(Y) = scale_ * EigenMap<float>(X);
  return Status::OK();
}

template <>
Status Erf<float>::Compute(OpKernelContext* context) const {
  auto X_ptr = context->Input<Tensor>(0);
  ORT_ENFORCE(X_ptr != nullptr);
  auto& X = *X_ptr;
  auto& Y = *context->Output(0, X.Shape());
  size_t n = (int64_t)X.Shape().Size();

#ifdef USE_OPENMP
  const int64_t block_size = 8192;
  if ((int64_t)n >= block_size*2) {
    int64_t blocks = ((int64_t)n + block_size - 1) / block_size;

    #pragma omp parallel for
    for (int64_t i = 0; i < blocks; ++i) {
      int64_t offset = i * block_size;
      int64_t remain = ((int64_t)n) - offset;
      int64_t len = std::min(remain, block_size);
      MlasComputeErff(X.Data<float>() + offset, Y.MutableData<float>() + offset, (size_t)len);
    }

    return Status::OK();
  }
#endif

  MlasComputeErff(X.Data<float>(), Y.MutableData<float>(), n);

  return Status::OK();
}


class Broadcast2Util {
public:
  static std::vector<int64_t> calcStrides(const std::vector<int64_t>& dims) {
    std::vector<int64_t> stride(dims.size(), 1LL);
    for (int64_t i = (int64_t)dims.size() - 2; i >= 0; --i) {
      stride[i] = stride[i+1] * dims[i+1];
    }
    return stride;
  }
  
  static void extend_prefix_with_1(std::vector<int64_t>& a, size_t num_axes) {
    if (a.size() < num_axes) {
      auto temp = std::vector<int64_t>(num_axes - a.size(), 1LL);
      temp.insert(temp.end(), a.begin(), a.end());
      a.swap(temp);
    }
  }
  
  static bool calcResultDims(const std::vector<int64_t>& a, const std::vector<int64_t>& b, std::vector<int64_t>& r) {
    r.clear();
    if (a.size() == 0 && b.size() == 0) return true;
  
    std::vector<int64_t> ta(a);
    std::vector<int64_t> tb(b);
    if (ta.size() == 0) ta.emplace_back(1LL);
    if (tb.size() == 0) tb.emplace_back(1LL);
    int64_t num_axes = (uint64_t)std::max(ta.size(), tb.size());
    extend_prefix_with_1(ta, num_axes);
    extend_prefix_with_1(tb, num_axes);
    
    for (int64_t i = 0; i < num_axes; ++i) {
      if (ta[i] == tb[i]) {
        r.push_back(ta[i]);
      }
      else if (ta[i] == 1LL) {
        r.push_back(tb[i]);
      }
      else if (tb[i] == 1LL) {
        r.push_back(ta[i]);
      }
      else {
        return false;
      }
    }
    return true;
  }
  
  // calc effective compact dims for broadcast2
  static bool calcCompactDims(const std::vector<int64_t>& a_orig, const std::vector<int64_t>& b_orig,
                              std::vector<int64_t>& a, std::vector<int64_t>& b, std::vector<int64_t>& r)
  {
    std::vector<int64_t> ta(a_orig);
    std::vector<int64_t> tb(b_orig);
    if (ta.size() == 0) ta.emplace_back(1LL);
    if (tb.size() == 0) tb.emplace_back(1LL);
    auto num_axes = std::max(ta.size(), tb.size());
    extend_prefix_with_1(ta, num_axes);
    extend_prefix_with_1(tb, num_axes);
    
    a.clear();
    b.clear();
    r.clear();
    for (int64_t i = (int64_t)num_axes - 1; i >= 0; --i) {
      if (ta[i] == 1LL && tb[i] == 1LL) continue;  // squeeze
      if (ta[i] != 1LL && tb[i] != 1LL && ta[i] != tb[i]) return false; //Can not broadcast
  
      if (a.size() > 0 &&
          ((a.back() == b.back() && ta[i] == tb[1]) ||
              (a.back() > b.back() && ta[i] > tb[i]) ||
              (a.back() < b.back() && ta[i] < tb[i]))) {
        a.back() *= ta[i];
        b.back() *= tb[i];
        continue;
      }
  
      if (a.size() > 0) r.emplace_back(std::max(a.back(), b.back()));
      a.emplace_back(ta[i]);
      b.emplace_back(tb[i]);
    }
    if (a.size() == 0) {
      a.emplace_back(1LL); b.emplace_back(1LL); r.emplace_back(1LL);
      return true;
    }
    r.emplace_back(std::max(a.back(), b.back()));
    std::reverse(a.begin(), a.end());
    std::reverse(b.begin(), b.end());
    std::reverse(r.begin(), r.end());
    return true;
  }
};


template<typename TIn, typename TOut>
class Broadcast2Operator {
public:
  // dimsA,  dimsB, dimsR is compacted by Broadcast2Util::calcCompactDims()
  Broadcast2Operator() = default;

  bool setInputOutput(const TIn* a, const std::vector<int64_t>& dimsA,
                      const TIn* b, const std::vector<int64_t>& dimsB, TOut* r) {
    a_ = a;
    b_ = b;
    r_ = r;
    std::vector<int64_t> da(dimsA);
    std::vector<int64_t> db(dimsB);
    if (!Broadcast2Util::calcCompactDims(dimsA, dimsB, da, db, dimsR_)) return false;
    num_axes_ = (int64_t)dimsR_.size();
    strideA_ = Broadcast2Util::calcStrides(da); 
    strideB_ = Broadcast2Util::calcStrides(db); 
    strideR_ = Broadcast2Util::calcStrides(dimsR_);
    target_size_ = 1LL;
    for (int64_t i = 0; i < (int64_t)dimsR_.size(); ++i) { 
      target_size_ *= dimsR_[i]; 
      if (da[i] == 1) strideA_[i] = 0LL;
      if (db[i] == 1) strideB_[i] = 0LL;
    }
    return true;
  }


  template<typename FuncScalaOpVec, typename FuncVecOpScala,  typename FuncVecOpVec>
  void run_split(int64_t start_row, int64_t num_rows, FuncScalaOpVec scalaOpVec, FuncVecOpScala vecOpScala, FuncVecOpVec vecOpVec) {
    std::vector<int64_t> idx(num_axes_, 0);
    std::vector<int64_t> dims(dimsR_);
    dims[0] = num_rows;

    TOut* r = r_ + start_row * strideR_[0];
    const TIn* b = b_ + start_row * strideA_[0];
    const TIn* a = a_ + start_row * strideB_[0];

    while (true) {
      if (strideB_.back() == 0LL) {
        vecOpScala(a, *b, r, dims.back());
      }
      else if (strideA_.back() != 0LL) {
        vecOpVec(a, b, r, dims.back());
      }
      else {
        scalaOpVec(*a, b, r, dims.back());
      }

      r += dims.back();
      int i = num_axes_ - 2;
      for ( ; i >= 0; --i) {
        a += strideA_[i];
        b += strideB_[i];
        if (++idx[i] < dims[i]) break;
        a -= (strideA_[i] * idx[i]);
        b -= (strideB_[i] * idx[i]);
        idx[i] = 0LL;
      }
      if (i < 0) break;
    } 
  }

  template<typename FuncScalaOpVec, typename FuncVecOpScala,  typename FuncVecOpVec>
  void execute_op(FuncScalaOpVec scalaOpVec, FuncVecOpScala vecOpScala, FuncVecOpVec vecOpVec) {

    int64_t num_splits = 1LL;
    int64_t rows_per_split = dimsR_[0];
    int64_t remain_rows = 0LL;

    #ifdef USE_OPENMP
    // always split at 0
    if (target_size_ >= 524288) {
      int64_t num_elems_per_row = strideR_[0];
      int64_t min_split_size = 2*65536; 
      int64_t max_splits = 48;
      int64_t min_rows_per_split = (min_split_size + (num_elems_per_row - 1)) / num_elems_per_row;
      num_splits = (dimsR_[0] + min_rows_per_split - 1) / min_rows_per_split;
      num_splits = std::min(max_splits, num_splits);
      rows_per_split = (dimsR_[0] + num_splits - 1) / num_splits;
      remain_rows = dimsR_[0] % num_splits;
    }

    #pragma omp parallel for
    #endif
    for (int64_t split = 0; split < num_splits; ++split) {
      int64_t start_row = (split < remain_rows) ? (split * (rows_per_split + 1)) : (remain_rows * (rows_per_split + 1) + (split - remain_rows)*rows_per_split);
      int64_t num_rows = (split < remain_rows) ? (rows_per_split + 1) : (rows_per_split);
      run_split(start_row, num_rows, scalaOpVec, vecOpScala, vecOpVec);
    }
  }

private:
  const TIn* a_;
  const TIn* b_;
  TOut* r_;
  std::vector<int64_t> dimsR_;

  int64_t num_axes_;
  int64_t target_size_;

  std::vector<int64_t> strideA_;
  std::vector<int64_t> strideB_;
  std::vector<int64_t> strideR_;
}; // Broadcast2Operator


class Broadcast2Wrapper {
public:
  template<typename TIn, typename TOut>
  static Status ExecuteOperator(OpKernelContext* context, 
                                std::function<void(const TIn value, const TIn* vec, TOut* out, int64_t num)> scalaOpVec,
                                std::function<void(const TIn* vec, const TIn value, TOut* out, int64_t num)> vecOpScala, 
                                std::function<void(const TIn* vec0, const TIn* vec1, TOut* out, int64_t num)> vecOpVec) {
    const Tensor* t0 = context->Input<Tensor>(0);
    const Tensor* t1 = context->Input<Tensor>(1);
    auto dims0 = t0->Shape().GetDims();
    auto dims1 = t1->Shape().GetDims();
    std::vector<int64_t> cd0, cd1, cdr, dr;
  
    if (!Broadcast2Util::calcResultDims(dims0, dims1, dr)) {
      throw 1;
    }
  
    Tensor* out = context->Output(0, dr);
    Broadcast2Operator<TIn, TOut> bo;
    bo.setInputOutput(t0->Data<TIn>(), dims0, t1->Data<TIn>(), dims1, out->MutableData<TOut>());
    bo.execute_op(scalaOpVec, vecOpScala, vecOpVec);
    return Status::OK();
  }
};

template <>
Status Mul<float>::Compute(OpKernelContext* context) const {
  return Broadcast2Wrapper::ExecuteOperator<float, float>(
    context, 
    [](const float v, const float* b, float* r, int64_t num) -> void {
        ConstEigenVectorMap<float> vb(b, num);
        EigenVectorMap<float> vr(r, num);
        vr = v * vb.array();
    }, 
    [](const float* a, const float v, float* r, int64_t num) {
        ConstEigenVectorMap<float> va(a, num);
        EigenVectorMap<float> vr(r, num);
        vr = va.array() * v;
    }, 
    [](const float* a, const float* b, float* r, int64_t num) {
        ConstEigenVectorMap<float> va(a, num);
        ConstEigenVectorMap<float> vb(b, num);
        EigenVectorMap<float> vr(r, num);
        vr = va.array() * vb.array();
    }
  );
}

template <>
Status Add<float>::Compute(OpKernelContext* context) const {
  return Broadcast2Wrapper::ExecuteOperator<float, float>(
    context, 
    [](const float v, const float* b, float* r, int64_t num) -> void {
        ConstEigenVectorMap<float> vb(b, num);
        EigenVectorMap<float> vr(r, num);
        vr = v + vb.array();
    }, 
    [](const float* a, const float v, float* r, int64_t num) {
        ConstEigenVectorMap<float> va(a, num);
        EigenVectorMap<float> vr(r, num);
        vr = va.array() + v;
    }, 
    [](const float* a, const float* b, float* r, int64_t num) {
        ConstEigenVectorMap<float> va(a, num);
        ConstEigenVectorMap<float> vb(b, num);
        EigenVectorMap<float> vr(r, num);
        vr = va.array() + vb.array();
    }
  );
}

template <>
Status Sub<float>::Compute(OpKernelContext* context) const {
  return Broadcast2Wrapper::ExecuteOperator<float, float>(
    context, 
    [](const float v, const float* b, float* r, int64_t num) -> void {
        ConstEigenVectorMap<float> vb(b, num);
        EigenVectorMap<float> vr(r, num);
        vr = v - vb.array();
    }, 
    [](const float* a, const float v, float* r, int64_t num) {
        ConstEigenVectorMap<float> va(a, num);
        EigenVectorMap<float> vr(r, num);
        vr = va.array() - v;
    }, 
    [](const float* a, const float* b, float* r, int64_t num) {
        ConstEigenVectorMap<float> va(a, num);
        ConstEigenVectorMap<float> vb(b, num);
        EigenVectorMap<float> vr(r, num);
        vr = va.array() - vb.array();
    }
  );
}

template <>
Status Div<float>::Compute(OpKernelContext* context) const {
  return Broadcast2Wrapper::ExecuteOperator<float, float>(
    context, 
    [](const float v, const float* b, float* r, int64_t num) -> void {
        ConstEigenVectorMap<float> vb(b, num);
        EigenVectorMap<float> vr(r, num);
        vr = v / vb.array();
    }, 
    [](const float* a, const float v, float* r, int64_t num) {
        ConstEigenVectorMap<float> va(a, num);
        EigenVectorMap<float> vr(r, num);
        vr = va.array() / v;
    }, 
    [](const float* a, const float* b, float* r, int64_t num) {
        ConstEigenVectorMap<float> va(a, num);
        ConstEigenVectorMap<float> vb(b, num);
        EigenVectorMap<float> vr(r, num);
        vr = va.array() / vb.array();
    }
  );
}


template <>
Status Pow<float>::Compute(OpKernelContext* context) const {
  return Broadcast2Wrapper::ExecuteOperator<float, float>(
    context, 
    [](const float& v, const float* b, float* r, int64_t num) -> void {
        ConstEigenVectorMap<float> vb(b, num);
        EigenVectorMap<float> vr(r, num);
        vr = Eigen::pow(v, vb.array());
    }, 
    [](const float* a, const float& v, float* r, int64_t num) {
        ConstEigenVectorMap<float> va(a, num);
        EigenVectorMap<float> vr(r, num);
        if (v == 2.0) {
          vr = Eigen::square(va.array());
        }
        else if (v == 3.0) {
          vr = Eigen::cube(va.array());
        }
        else {
          vr = Eigen::pow(va.array(), v);
        }
    }, 
    [](const float* a, const float* b, float* r, int64_t num) {
        ConstEigenVectorMap<float> va(a, num);
        ConstEigenVectorMap<float> vb(b, num);
        EigenVectorMap<float> vr(r, num);
        vr = Eigen::pow(va.array(), vb.array());
    }
  );
}

} // namespace onnxruntime

