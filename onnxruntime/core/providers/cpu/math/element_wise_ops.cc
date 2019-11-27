// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/math/element_wise_ops.h"
#include <unsupported/Eigen/SpecialFunctions>
#include "core/util/math.h"
#include "core/mlas/inc/mlas.h"

#include <cmath>

namespace onnxruntime {

#define REG_ELEMENTWISE_TYPED_KERNEL(OP_TYPE, VERSION, TYPE, KERNEL_CLASS)         \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                                  \
      OP_TYPE,                                                                     \
      VERSION,                                                                     \
      TYPE,                                                                        \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<TYPE>()), \
      KERNEL_CLASS<TYPE>);

#define REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(OP_TYPE, VERSION, TYPE, KERNEL_CLASS) \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                                    \
      OP_TYPE,                                                                       \
      VERSION,                                                                       \
      TYPE,                                                                          \
      KernelDefBuilder()                                                             \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<TYPE>())                  \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<bool>()),                \
      KERNEL_CLASS<TYPE>);

#define REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(OP_TYPE, VERSION_FROM, VERSION_TO, TYPE, KERNEL_CLASS) \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                                                           \
      OP_TYPE,                                                                                        \
      VERSION_FROM, VERSION_TO,                                                                       \
      TYPE,                                                                                           \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<TYPE>()),                    \
      KERNEL_CLASS<TYPE>);

#define REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(OP_TYPE, VERSION_FROM, VERSION_TO, TYPE, KERNEL_CLASS) \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                                                                     \
      OP_TYPE,                                                                                                  \
      VERSION_FROM, VERSION_TO,                                                                                 \
      TYPE,                                                                                                     \
      KernelDefBuilder()                                                                                        \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<TYPE>())                                             \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<bool>()),                                           \
      KERNEL_CLASS<TYPE>);

REG_ELEMENTWISE_TYPED_KERNEL(Add, 7, float, Add);
REG_ELEMENTWISE_TYPED_KERNEL(Add, 7, double, Add);
REG_ELEMENTWISE_TYPED_KERNEL(Add, 7, int32_t, Add);
REG_ELEMENTWISE_TYPED_KERNEL(Add, 7, int64_t, Add);

REG_ELEMENTWISE_TYPED_KERNEL(Sub, 7, float, Sub);
REG_ELEMENTWISE_TYPED_KERNEL(Sub, 7, double, Sub);
REG_ELEMENTWISE_TYPED_KERNEL(Sub, 7, int32_t, Sub);
REG_ELEMENTWISE_TYPED_KERNEL(Sub, 7, int64_t, Sub);

REG_ELEMENTWISE_TYPED_KERNEL(Mul, 7, float, Mul);
REG_ELEMENTWISE_TYPED_KERNEL(Mul, 7, double, Mul);
REG_ELEMENTWISE_TYPED_KERNEL(Mul, 7, int32_t, Mul);
REG_ELEMENTWISE_TYPED_KERNEL(Mul, 7, int64_t, Mul);

REG_ELEMENTWISE_TYPED_KERNEL(Div, 7, float, Div);
REG_ELEMENTWISE_TYPED_KERNEL(Div, 7, double, Div);
REG_ELEMENTWISE_TYPED_KERNEL(Div, 7, int32_t, Div);
REG_ELEMENTWISE_TYPED_KERNEL(Div, 7, int64_t, Div);

REG_ELEMENTWISE_TYPED_KERNEL(Abs, 6, float, Abs);
REG_ELEMENTWISE_TYPED_KERNEL(Abs, 6, double, Abs);
REG_ELEMENTWISE_TYPED_KERNEL(Abs, 6, int8_t, Abs);
REG_ELEMENTWISE_TYPED_KERNEL(Abs, 6, int16_t, Abs);
REG_ELEMENTWISE_TYPED_KERNEL(Abs, 6, int32_t, Abs);
REG_ELEMENTWISE_TYPED_KERNEL(Abs, 6, int64_t, Abs);
REG_ELEMENTWISE_TYPED_KERNEL(Abs, 6, uint8_t, Abs);
REG_ELEMENTWISE_TYPED_KERNEL(Abs, 6, uint16_t, Abs);
REG_ELEMENTWISE_TYPED_KERNEL(Abs, 6, uint32_t, Abs);
REG_ELEMENTWISE_TYPED_KERNEL(Abs, 6, uint64_t, Abs);

REG_ELEMENTWISE_TYPED_KERNEL(Neg, 6, float, Neg);
REG_ELEMENTWISE_TYPED_KERNEL(Neg, 6, double, Neg);
REG_ELEMENTWISE_TYPED_KERNEL(Neg, 6, int8_t, Neg);
REG_ELEMENTWISE_TYPED_KERNEL(Neg, 6, int32_t, Neg);

REG_ELEMENTWISE_TYPED_KERNEL(Floor, 6, float, Floor);

REG_ELEMENTWISE_TYPED_KERNEL(Ceil, 6, float, Ceil);

REG_ELEMENTWISE_TYPED_KERNEL(Reciprocal, 6, float, Reciprocal);

REG_ELEMENTWISE_TYPED_KERNEL(Sqrt, 6, float, Sqrt);
REG_ELEMENTWISE_TYPED_KERNEL(Sqrt, 6, double, Sqrt);

REG_ELEMENTWISE_TYPED_KERNEL(Pow, 7, float, Pow);
REG_ELEMENTWISE_TYPED_KERNEL(Pow, 7, double, Pow);

REG_ELEMENTWISE_TYPED_KERNEL(Exp, 6, float, Exp);
REG_ELEMENTWISE_TYPED_KERNEL(Exp, 6, double, Exp);

REG_ELEMENTWISE_TYPED_KERNEL(Log, 6, float, Log);

REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Sum, 6, 7, float, Sum_6);
REG_ELEMENTWISE_TYPED_KERNEL(Sum, 8, float, Sum_8);

REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Min, 6, 7, float, Min_6);
REG_ELEMENTWISE_TYPED_KERNEL(Min, 8, float, Min_8);

REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Max, 6, 7, float, Max_6);
REG_ELEMENTWISE_TYPED_KERNEL(Max, 8, float, Max_8);
REG_ELEMENTWISE_TYPED_KERNEL(Max, 8, double, Max_8);

REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Less, 7, 9, float, Less);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Less, 7, 9, double, Less);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Less, 9, int32_t, Less);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Less, 9, int64_t, Less);

REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Greater, 7, 9, float, Greater)
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Greater, 9, int32_t, Greater);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Greater, 9, int64_t, Greater);

REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Equal, 7, 10, bool, Equal);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Equal, 7, 10, int32_t, Equal);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Equal, 7, 10, int64_t, Equal);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Equal, 11, bool, Equal);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Equal, 11, int32_t, Equal);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Equal, 11, int64_t, Equal);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Equal, 11, float, Equal);

REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Mean, 6, 7, float, Mean_6);
REG_ELEMENTWISE_TYPED_KERNEL(Mean, 8, float, Mean_8);

REG_ELEMENTWISE_TYPED_KERNEL(BitShift, 11, uint8_t, BitShift);
//REG_ELEMENTWISE_TYPED_KERNEL(BitShift, 11, uint16_t, BitShift);
REG_ELEMENTWISE_TYPED_KERNEL(BitShift, 11, uint32_t, BitShift);
REG_ELEMENTWISE_TYPED_KERNEL(BitShift, 11, uint64_t, BitShift);

REG_ELEMENTWISE_TYPED_KERNEL(Erf, 9, float, Erf);

// REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Not, 1, bool, Not);
// REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(And, 7, bool, And);
// REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Or, 7, bool, Or);
// REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Xor, 7, bool, Xor);

ONNX_CPU_OPERATOR_KERNEL(
    Not,
    1,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<bool>())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<bool>()),
    Not);

ONNX_CPU_OPERATOR_KERNEL(
    And,
    7,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<bool>())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<bool>()),
    And);

ONNX_CPU_OPERATOR_KERNEL(
    Or,
    7,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<bool>())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<bool>()),
    Or);

ONNX_CPU_OPERATOR_KERNEL(
    Xor,
    7,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<bool>())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<bool>()),
    Xor);

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

template <typename T>
Status Sqrt<T>::Compute(OpKernelContext* ctx) const {
  auto& X = *ctx->Input<Tensor>(0);
  auto& Y = *ctx->Output(0, X.Shape());

  EigenMap<T>(Y) = EigenMap<T>(X).cwiseSqrt();

  return Status::OK();
}

template <typename T>
Status Pow<T>::Compute(OpKernelContext* context) const {
  const Tensor& Y = *context->Input<Tensor>(1);
  std::function<void(EigenVectorMap<T>, ConstEigenVectorMap<T>, T)> input1scalar =
      [](EigenVectorMap<T> output, ConstEigenVectorMap<T> input0, T input1) { output = Eigen::pow(input0.array(), input1); };
  if (Y.Shape().Size() == 1) {
    T value = *Y.Data<T>();
    if (value == 2.0) {
      input1scalar = [](EigenVectorMap<T> output, ConstEigenVectorMap<T> input0, T) { output = Eigen::square(input0.array()); };
    } else if (value == 3.0) {
      input1scalar = [](EigenVectorMap<T> output, ConstEigenVectorMap<T> input0, T) { output = Eigen::cube(input0.array()); };
    }
  }

  return BroadcastTwo<T, T>(
      *context,
      [](EigenVectorMap<T> output, T input0, ConstEigenVectorMap<T> input1) { output = Eigen::pow(input0, input1.array()); },
      input1scalar,
      [](EigenVectorMap<T> output, ConstEigenVectorMap<T> input0, ConstEigenVectorMap<T> input1) { output = Eigen::pow(input0.array(), input1.array()); });
}

template <typename T>
Status Exp<T>::Compute(OpKernelContext* ctx) const {
  auto& X = *ctx->Input<Tensor>(0);
  auto& Y = *ctx->Output(0, X.Shape());

  EigenMap<T>(Y) = EigenMap<T>(X).array().exp();

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

template <typename T>
Status Max_8<T>::Compute(OpKernelContext* context) const {
  return BroadcastVariadic<T, T>(
      Node(), *context,
      [](EigenVectorMap<T> output, T input0, ConstEigenVectorMap<T> input1) { output = input1.array().max(input0); },
      [](EigenVectorMap<T> output, ConstEigenVectorMap<T> input0, T input1) { output = input0.array().max(input1); },
      [](EigenVectorMap<T> output, ConstEigenVectorMap<T> input0, ConstEigenVectorMap<T> input1) { output = input0.array().max(input1.array()); });
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

template <typename T>
BitShift<T>::BitShift(const OpKernelInfo& info) : OpKernel(info) {
  std::string direction;
  auto status = info.GetAttr("direction", &direction);
  ORT_ENFORCE(status.IsOK(), status);

  if (direction == "LEFT")
    shift_left_ = true;
  else if (direction == "RIGHT")
    shift_left_ = false;
  else
    ORT_THROW("Invalid direction value of '", direction, "'. Valid values are 'LEFT' or 'RIGHT'.");
}

template <typename T>
Status BitShift<T>::Compute(OpKernelContext* context) const {
  return BroadcastTwo<T, T>(
      *context,
      [this](EigenVectorMap<T> output, T input0, ConstEigenVectorMap<T> input1) {
        int64_t i = 0;
        if (shift_left_) {
          for (const auto& input : input1.array()) {
            output[i++] = input0 << input;
          }
        } else {
          for (const auto& input : input1.array()) {
            output[i++] = input0 >> input;
          }
        }
      },
      [this](EigenVectorMap<T> output, ConstEigenVectorMap<T> input0, T input1) {
        int64_t i = 0;
        if (shift_left_) {
          for (const auto& input : input0.array()) {
            output[i++] = input << input1;
          }
        } else {
          for (const auto& input : input0.array()) {
            output[i++] = input >> input1;
          }
        }
      },
      [this](EigenVectorMap<T> output, ConstEigenVectorMap<T> input0, ConstEigenVectorMap<T> input1) {
        auto cur0 = input0.begin(), end0 = input0.end();
        auto cur1 = input1.begin(), end1 = input1.end();
        auto cur_out = output.begin(), end_out = output.end();
        if (shift_left_) {
          for (; cur0 != end0; ++cur0, ++cur1, ++cur_out) {
            *cur_out = *cur0 << *cur1;
          }
        } else {
          for (; cur0 != end0; ++cur0, ++cur1, ++cur_out) {
            *cur_out = *cur0 >> *cur1;
          }
        }

        ORT_ENFORCE(cur1 == end1);
        ORT_ENFORCE(cur_out == end_out);
      });
}

template <typename T>
class Sin final : public OpKernel {
 public:
  Sin(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    auto& X = *context->Input<Tensor>(0);
    auto& Y = *context->Output(0, X.Shape());
    MakeEigenArrayMap<T>(Y) = MakeEigenArrayMap<T>(X).sin();
    return Status::OK();
  }
};

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Sin,
    7,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Sin<float>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Sin,
    7,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    Sin<double>);

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

    for (size_t index = 0; index < in.size(); ++index) {
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

    for (size_t index = 0; index < in.size(); ++index) {
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

    for (size_t index = 0; index < in.size(); ++index) {
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
  const auto* p_shape = tensor_shape.template Data<int64_t>();
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

#define REG_EXPAND_KERNEL(TYPE)                                                    \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                                  \
      Expand,                                                                      \
      8,                                                                           \
      TYPE,                                                                        \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<TYPE>()), \
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
Status Erf<float>::Compute(OpKernelContext* context) const {
  auto X_ptr = context->Input<Tensor>(0);
  ORT_ENFORCE(X_ptr != nullptr);
  auto& X = *X_ptr;
  auto& Y = *context->Output(0, X.Shape());

  MlasComputeErf(X.template Data<float>(), Y.template MutableData<float>(), X.Shape().Size());

  return Status::OK();
}

class Mod final : public OpKernel {
 public:
  Mod(const OpKernelInfo& info) : OpKernel(info) {
    int64_t fmod = 0;
    Status s = info.GetAttr<int64_t>("fmod", &fmod);
    if (s.IsOK()) {
      ORT_ENFORCE((fmod == 0) || (fmod == 1), "fmod must have value either 0 or 1");
      fmod_ = (fmod == 1);
    }
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  bool fmod_{false};
};

ONNX_CPU_OPERATOR_KERNEL(
    Mod,
    10,
    KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                                            DataTypeImpl::GetTensorType<double>(),
                                            DataTypeImpl::GetTensorType<int64_t>(),
                                            DataTypeImpl::GetTensorType<uint64_t>(),
                                            DataTypeImpl::GetTensorType<int32_t>(),
                                            DataTypeImpl::GetTensorType<uint32_t>(),
                                            DataTypeImpl::GetTensorType<int16_t>(),
                                            DataTypeImpl::GetTensorType<uint16_t>(),
                                            DataTypeImpl::GetTensorType<int8_t>(),
                                            DataTypeImpl::GetTensorType<uint8_t>(),
                                            DataTypeImpl::GetTensorType<MLFloat16>()}),
    Mod);

namespace mod_internal {

template <class T>
void BroadCastFMod(const Tensor& X, const Tensor& Y, OpKernelContext* context) {
  TBroadcaster<T, T> mod_broadcaster{X, Y};
  Tensor* const output = context->Output(0, mod_broadcaster.GetOutputShape());
  ORT_ENFORCE(output, "failed to get first output!");
  TBroadcastOutput<T> mod_broadcast_output{
      mod_broadcaster.GetSpanSize(), *output};

  BroadcastLoopSpan(
      mod_broadcaster, mod_broadcast_output,
      [](gsl::span<T> output, const T& X, gsl::span<const T> Y) {
        std::transform(Y.cbegin(), Y.cend(), output.begin(),
                       [X](T y) {
                         return static_cast<T>(std::fmod(X, y));
                       });
      },
      [](gsl::span<T> output, gsl::span<const T> X, const T& Y) {
        std::transform(X.cbegin(), X.cend(), output.begin(),
                       [Y](T x) {
                         return static_cast<T>(std::fmod(x, Y));
                       });
      },
      [](gsl::span<T> output, gsl::span<const T> X, gsl::span<const T> Y) {
        std::transform(
            X.cbegin(), X.cend(), Y.cbegin(), output.begin(),
            [](T x, T y) {
              return static_cast<T>(std::fmod(x, y));
            });
      });
}

template <class T>
inline T Modulus(T x, T y) {
  auto res = x % y;
  if ((res < 0 && y > 0) || (res > 0 && y < 0)) {
    res += y;
  }
  return static_cast<T>(res);
}

template <class T>
void BroadCastMod(const Tensor& X, const Tensor& Y, OpKernelContext* context) {
  TBroadcaster<T, T> mod_broadcaster{X, Y};
  Tensor* const output = context->Output(0, mod_broadcaster.GetOutputShape());
  ORT_ENFORCE(output, "failed to get first output!");
  TBroadcastOutput<T> mod_broadcast_output{
      mod_broadcaster.GetSpanSize(), *output};

  // static_cast below are necessary when small types such as
  // int16_t and int8_t are converted to integers to perform remainder
  // operation. This cast is safe with respect to data loss.
  BroadcastLoopSpan(
      mod_broadcaster, mod_broadcast_output,
      [](gsl::span<T> output, const T& X, gsl::span<const T> Y) {
        std::transform(Y.cbegin(), Y.cend(), output.begin(),
                       [X](T y) {
                         return Modulus(X, y);
                       });
      },
      [](gsl::span<T> output, gsl::span<const T> X, const T& Y) {
        std::transform(X.cbegin(), X.cend(), output.begin(),
                       [Y](T x) {
                         return Modulus(x, Y);
                       });
      },
      [](gsl::span<T> output, gsl::span<const T> X, gsl::span<const T> Y) {
        std::transform(
            X.cbegin(), X.cend(), Y.cbegin(), output.begin(),
            [](T x, T y) {
              return Modulus(x, y);
            });
      });
}

void BroadCastMFloat16FMod(const Tensor& X, const Tensor& Y, OpKernelContext* context) {
  TBroadcaster<MLFloat16, MLFloat16> mod_broadcaster{X, Y};
  Tensor* const output = context->Output(0, mod_broadcaster.GetOutputShape());
  ORT_ENFORCE(output, "failed to get first output!");
  TBroadcastOutput<MLFloat16> mod_broadcast_output{
      mod_broadcaster.GetSpanSize(), *output};

  BroadcastLoopSpan(
      mod_broadcaster, mod_broadcast_output,
      [](gsl::span<MLFloat16> output, const MLFloat16& X, gsl::span<const MLFloat16> Y) {
        std::transform(Y.cbegin(), Y.cend(), output.begin(),
                       [X_fl = math::halfToFloat(X.val)](const MLFloat16& y) {
                         return MLFloat16(math::floatToHalf(std::fmod(X_fl, math::halfToFloat(y.val))));
                       });
      },
      [](gsl::span<MLFloat16> output, gsl::span<const MLFloat16> X, const MLFloat16& Y) {
        std::transform(X.cbegin(), X.cend(), output.begin(),
                       [Y_fl = math::halfToFloat(Y.val)](const MLFloat16& x) {
                         return MLFloat16(math::floatToHalf(std::fmod(math::halfToFloat(x.val), Y_fl)));
                       });
      },
      [](gsl::span<MLFloat16> output, gsl::span<const MLFloat16> X, gsl::span<const MLFloat16> Y) {
        std::transform(
            X.cbegin(), X.cend(), Y.cbegin(), output.begin(),
            [](const MLFloat16& x, const MLFloat16& y) {
              auto x_fl = math::halfToFloat(x.val);
              auto y_fl = math::halfToFloat(y.val);
              return MLFloat16(math::floatToHalf(std::fmod(x_fl, y_fl)));
            });
      });
}

// Generic implementation of Mod kernel
template <class T>
struct CallModImpl {
  void operator()(bool fmod, const Tensor& X, const Tensor& Y, OpKernelContext* ctx) const {
    if (fmod) {
      BroadCastFMod<T>(X, Y, ctx);
    } else {
      BroadCastMod<T>(X, Y, ctx);
    }
  }
};

}  // namespace mod_internal

Status Mod::Compute(OpKernelContext* context) const {
  Status s;

  const auto& X = *context->Input<Tensor>(0);
  const auto& Y = *context->Input<Tensor>(1);

  auto dtype = X.DataType();
  if (dtype != Y.DataType()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "X and Y input types do not match: ",
                           dtype, " vs ", Y.DataType());
  }

  using namespace mod_internal;

  namespace on = ONNX_NAMESPACE;
  auto dt_type = X.GetElementType();
  switch (dt_type) {
    case on::TensorProto_DataType_FLOAT:
      ORT_ENFORCE(fmod_, "fmod attribute must be true for float, float16 and double types");
      BroadCastFMod<float>(X, Y, context);
      break;
    case on::TensorProto_DataType_DOUBLE:
      ORT_ENFORCE(fmod_, "fmod attribute must be true for float, float16 and double types");
      BroadCastFMod<double>(X, Y, context);
      break;
    case on::TensorProto_DataType_FLOAT16:
      ORT_ENFORCE(fmod_, "fmod attribute must be true for float, float16 and double types");
      BroadCastMFloat16FMod(X, Y, context);
      break;
    default:
      utils::MLTypeCallDispatcher<mod_internal::CallModImpl, uint8_t, int8_t, uint16_t, int16_t,
                                  uint32_t, int32_t, uint64_t, int64_t>
          t_disp(dt_type);
      t_disp.Invoke(fmod_, X, Y, context);
      break;
  }
  return s;
}

}  // namespace onnxruntime
