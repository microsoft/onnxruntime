// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/math/element_wise_ops.h"

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

template <typename T>
auto MakeEigenArrayMap(Tensor& t) { return EigenVectorArrayMap<T>(t.template MutableData<T>(), t.Shape().Size()); }
template <typename T>
auto MakeEigenArrayMap(const Tensor& t) { return ConstEigenVectorArrayMap<T>(t.template Data<T>(), t.Shape().Size()); }

struct BroadcastIterator {
  size_t AdvanceBy(size_t delta) {
    size_t index = index_;

    index_ += deltas_[0] * delta;
    counters_[0] += delta;
    if (counters_[0] == counts_[0]) {
      counters_[0] = 0;
      for (size_t counterIndex = 1; counterIndex < counters_.size(); counterIndex++) {
        index_ += deltas_[counterIndex];
        if (++counters_[counterIndex] != counts_[counterIndex])
          break;
        counters_[counterIndex] = 0;
      }
    }
    return index;
  }

  void Init(int64_t axis, int64_t largest) {
    ONNXRUNTIME_ENFORCE(axis == 1 || axis == largest, "Attempting to broadcast an axis by a dimension other than 1. ", axis, " by ", largest);

    deltas_.push_back(axis > 1);
    counts_.push_back(largest);
    count_ *= axis;
  }

  void Append(int64_t axis, int64_t largest) {
    ONNXRUNTIME_ENFORCE(axis == 1 || axis == largest, "Attempting to broadcast an axis by a dimension other than 1. ", axis, " by ", largest);

    // If we're greater than 1, it doesn't matter what the other tensor does
    if (axis > 1) {
      if (deltas_.back() <= 0)  // Were we broadcasting
        StopBroadcasting();
    } else {  // We must be 1, at this point
      if (deltas_.back() > 0)
        StartBroadcasting();
    }

    counts_.back() *= largest;  // Just increase the last count
    count_ *= axis;
  }

  void StopBroadcasting() {
    deltas_.push_back(count_);
    counts_.push_back(1);
  }

  void StartBroadcasting() {
    deltas_.push_back(-count_);
    counts_.push_back(1);
  }

  std::vector<int64_t> counters_;
  std::vector<ptrdiff_t> deltas_;
  std::vector<int64_t> counts_;
  size_t count_{1};  // Running total count of entries in tensor, used while building up the entries

 private:
  size_t index_{};
};

struct Broadcaster {
  Broadcaster(const std::vector<int64_t>& shape1, const std::vector<int64_t>& shape2) {
    size_t dimension_count_max = std::max(shape1.size(), shape2.size());
    size_t dimension_count_min = std::min(shape1.size(), shape2.size());
    output_shape_.resize(dimension_count_max);

    auto iter1 = shape1.end();
    auto iter2 = shape2.end();
    auto output_shape = output_shape_.end();

    // Scalars are a special case, as it's always a broadcast
    size_t index = 0;
    if (dimension_count_min == 0) {
      if (shape1.size() == 0)  // Shape1 is a scalar
      {
        if (shape2.size() == 0)  // Two scalars?
        {
          iterator1_.Init(1, 1);
          iterator2_.Init(1, 1);
        } else {
          auto axis = *--iter2;
          iterator1_.Init(1, axis);
          iterator2_.Init(axis, axis);
          *--output_shape = axis;
        }
      } else {  // Shape2 is a scalar
        auto axis = *--iter1;
        iterator1_.Init(axis, axis);
        iterator2_.Init(1, axis);
        *--output_shape = axis;
      }
      index++;  // Manually increment since we processed one axis
    }

    for (; index < dimension_count_min; index++) {
      auto axis1 = *--iter1;
      auto axis2 = *--iter2;

      auto largest = std::max(axis1, axis2);
      *--output_shape = largest;

      if (largest == 1 && index + 1 < dimension_count_min)  // Nothing to do in this case
        continue;

      iterator1_.Init(axis1, largest);
      iterator2_.Init(axis2, largest);
      index++;  // Manually increment since we processed one axis
      break;
    }

    for (; index < dimension_count_min; index++) {
      auto axis1 = *--iter1;
      auto axis2 = *--iter2;

      auto largest = std::max(axis1, axis2);
      *--output_shape = largest;

      if (largest == 1)  // Nothing to do in this case
        continue;

      iterator1_.Append(axis1, largest);
      iterator2_.Append(axis2, largest);
    }

    // If one shape is bigger than another we need to broadcast the smaller onto the bigger from this point on
    for (; index < dimension_count_max; index++) {
      if (dimension_count_max == shape2.size()) {
        auto axis = *--iter2;
        iterator1_.Append(1, axis);
        iterator2_.Append(axis, axis);
        *--output_shape = axis;
      } else {
        auto axis = *--iter1;
        iterator1_.Append(axis, axis);
        iterator2_.Append(1, axis);
        *--output_shape = axis;
      }
    }

    // Allocate the counters
    iterator1_.counters_.resize(iterator1_.counts_.size(), 0);
    iterator2_.counters_.resize(iterator2_.counts_.size(), 0);
  }

  size_t GetSpanSize() const { return std::min(iterator1_.counts_.front(), iterator2_.counts_.front()); }

  BroadcastIterator iterator1_, iterator2_;
  std::vector<int64_t> output_shape_;
};

template <typename T>
struct TBroadcaster {
  TBroadcaster(const Tensor& input0, const Tensor& input1)
      : input_tensor0_(input0),
        input_tensor1_(input1) {
  }

  TensorShape GetOutputShape() const { return TensorShape(broadcaster_.output_shape_); }
  size_t GetSpanSize() const { return span_size_; }

  bool IsInput0Scalar() const { return broadcaster_.iterator1_.deltas_.front() == 0; }
  bool IsInput1Scalar() const { return broadcaster_.iterator2_.deltas_.front() == 0; }

  T NextScalar0() { return *Next0(); }
  T NextScalar1() { return *Next1(); }

  gsl::span<const T> NextSpan0() { return gsl::span<const T>(Next0(), span_size_); }
  gsl::span<const T> NextSpan1() { return gsl::span<const T>(Next1(), span_size_); }

  ConstEigenVectorMap<T> NextEigen0() { return ConstEigenVectorMap<T>(Next0(), span_size_); }
  ConstEigenVectorMap<T> NextEigen1() { return ConstEigenVectorMap<T>(Next1(), span_size_); }

 private:
  const T* Next0() { return input0_ + broadcaster_.iterator1_.AdvanceBy(span_size_); }
  const T* Next1() { return input1_ + broadcaster_.iterator2_.AdvanceBy(span_size_); }

  const Tensor& input_tensor0_;
  const Tensor& input_tensor1_;
  Broadcaster broadcaster_{input_tensor0_.Shape().GetDims(), input_tensor1_.Shape().GetDims()};
  size_t span_size_{broadcaster_.GetSpanSize()};

  const T* input0_{input_tensor0_.template Data<T>()};
  const T* input1_{input_tensor1_.template Data<T>()};
};

template <typename T>
struct TBroadcastOutput {
  TBroadcastOutput(size_t span_size, Tensor& tensor)
      : span_size_(span_size) {
    output_ = tensor.template MutableData<T>();
    output_end_ = output_ + tensor.Shape().Size();
  }

  operator bool() const {
    return output_ != output_end_;
  }

  EigenVectorMap<T> NextEigenOutput() {
    return EigenVectorMap<T>(NextOutput(), span_size_);
  }

  gsl::span<T> NextSpanOutput() {
    return gsl::span<T>(NextOutput(), span_size_);
  }

 private:
  T* NextOutput() {
    T* output = output_;
    output_ += span_size_;
    return output;
  }

  T* output_;
  const T* output_end_;
  size_t span_size_;
};

template <typename T>
struct TensorAllocator {
  TensorAllocator(OpKernelContext& context) {
    ONNXRUNTIME_ENFORCE(context.GetTempSpaceAllocator(&allocator_).IsOK());
  }

  std::unique_ptr<Tensor> Allocate(const TensorShape& shape) {
    return std::make_unique<Tensor>(DataTypeImpl::GetType<T>(),
                                    shape,
                                    allocator_->Alloc(sizeof(T) * shape.Size()),
                                    allocator_->Info(),
                                    allocator_);
  }

 private:
  AllocatorPtr allocator_;
};

// Broadcast loop for when using eigen, functions are in this form:
// Input0Scalar: [](EigenVectorMap<T> output, T input0, ConstEigenVectorMap<T> input1)
// Input1Scalar: [](EigenVectorMap<T> output, ConstEigenVectorMap<T> input0, T input1)
// General     : [](EigenVectorMap<T> output, ConstEigenVectorMap<T> input0, ConstEigenVectorMap<T> input1)
template <typename TBroadcaster, typename Output, typename Input0Scalar, typename Input1Scalar, typename General>
void BroadcastLoop(TBroadcaster& bc, Output& output, Input0Scalar input0scalar, Input1Scalar input1scalar, General general) {
  if (bc.IsInput0Scalar()) {
    while (output)
      input0scalar(output.NextEigenOutput(), bc.NextScalar0(), bc.NextEigen1());
  } else if (bc.IsInput1Scalar()) {
    while (output)
      input1scalar(output.NextEigenOutput(), bc.NextEigen0(), bc.NextScalar1());
  } else {
    while (output)
      general(output.NextEigenOutput(), bc.NextEigen0(), bc.NextEigen1());
  }
}

template <typename TInput, typename TOutput, typename Input0Scalar, typename Input1Scalar, typename General>
Status BroadcastTwo(OpKernelContext& context, Input0Scalar input0scalar, Input1Scalar input1scalar, General general) {
  TBroadcaster<TInput> bc(*context.Input<Tensor>(0), *context.Input<Tensor>(1));
  TBroadcastOutput<TOutput> output(bc.GetSpanSize(), *context.Output(0, bc.GetOutputShape()));
  BroadcastLoop(bc, output, input0scalar, input1scalar, general);

  return Status::OK();
}

template <typename TInput, typename TOutput, typename Input0Scalar, typename Input1Scalar, typename General>
Status BroadcastVariadic(const Node& node, OpKernelContext& context, Input0Scalar input0scalar, Input1Scalar input1scalar, General general) {
  auto input_count = node.InputArgCount().front();
  ONNXRUNTIME_ENFORCE(input_count >= 1, "Must have 1 or more inputs");

  // One item is trivial, just copy across and exit
  if (input_count == 1) {
    EigenMap<TOutput>(*context.Output(0, context.Input<Tensor>(0)->Shape())) = EigenMap<TInput>(*context.Input<Tensor>(0));
    return Status::OK();
  }

  std::unique_ptr<Tensor> tempInput;
  std::unique_ptr<Tensor> tempOutput;

  TensorAllocator<TOutput> tensorAllocator(context);

  // For more than 2 tensors, we sum the first two into a temporary tensor, then sum the next with the temporary tensor
  for (int i = 0; i < input_count - 1; i++) {
    auto& tensor0 = tempInput ? *tempInput : *context.Input<Tensor>(0);
    auto& tensor1 = *context.Input<Tensor>(i + 1);

    TBroadcaster<TInput> bc(tensor0, tensor1);

    // Create a temporary output for all but the last iteration, which goes to the real output
    Tensor* p_output{};
    if (i == input_count - 2)
      p_output = context.Output(0, bc.GetOutputShape());
    else {
      tempOutput = tensorAllocator.Allocate(bc.GetOutputShape());
      p_output = tempOutput.get();
    }

    TBroadcastOutput<TOutput> output(bc.GetSpanSize(), *p_output);

    BroadcastLoop(bc, output, input0scalar, input1scalar, general);

    tempInput = std::move(tempOutput);
  }
  return Status::OK();
}

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

template <>
Status Pow<float>::Compute(OpKernelContext* context) const {
  return BroadcastTwo<float, float>(
      *context,
      [](EigenVectorMap<float> output, float input0, ConstEigenVectorMap<float> input1) { output = Eigen::pow(input0, input1.array()); },
      [](EigenVectorMap<float> output, ConstEigenVectorMap<float> input0, float input1) { output = Eigen::pow(input0.array(), input1); },
      [](EigenVectorMap<float> output, ConstEigenVectorMap<float> input0, ConstEigenVectorMap<float> input1) { output = Eigen::pow(input0.array(), input1.array()); });
}

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
  ONNXRUNTIME_ENFORCE(input_count >= 1, "Must have 1 or more inputs");
  auto& data_0 = *ctx->Input<Tensor>(0);
  auto& shape = data_0.Shape();
  auto sum = EigenMap<float>(*ctx->Output(0, shape));

  if (input_count == 1) {
    sum = EigenMap<float>(data_0);
  } else {
    auto& data_1 = *ctx->Input<Tensor>(1);
    ONNXRUNTIME_ENFORCE(data_1.Shape() == shape, "All inputs must have the same shape");

    sum = EigenMap<float>(data_0) + EigenMap<float>(data_1);
    for (int index = 2; index < input_count; index++) {
      auto& data_n = *ctx->Input<Tensor>(index);
      ONNXRUNTIME_ENFORCE(data_n.Shape() == shape, "All inputs must have the same shape");
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
  ONNXRUNTIME_ENFORCE(inputCount >= 1, "Must have 1 or more inputs");
  auto& data_0 = *ctx->Input<Tensor>(0);
  auto& shape = data_0.Shape();
  auto min = EigenMap<float>(*ctx->Output(0, shape));

  min = EigenMap<float>(data_0);
  for (int index = 1; index < inputCount; index++) {
    auto& data_n = *ctx->Input<Tensor>(index);
    ONNXRUNTIME_ENFORCE(data_n.Shape() == shape, "All inputs must have the same shape");
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
  ONNXRUNTIME_ENFORCE(inputCount >= 1, "Must have 1 or more inputs");
  auto& data_0 = *ctx->Input<Tensor>(0);
  auto& shape = data_0.Shape();
  auto max = EigenMap<float>(*ctx->Output(0, shape));

  max = EigenMap<float>(data_0);
  for (int index = 1; index < inputCount; index++) {
    auto& data_n = *ctx->Input<Tensor>(index);
    ONNXRUNTIME_ENFORCE(data_n.Shape() == shape, "All inputs must have the same shape");
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
  ONNXRUNTIME_ENFORCE(inputCount >= 1, "Must have 1 or more inputs");
  auto& data_0 = *ctx->Input<Tensor>(0);
  auto& shape = data_0.Shape();
  auto mean = EigenMap<float>(*ctx->Output(0, shape));

  if (inputCount == 1) {
    mean = EigenMap<float>(data_0);
  } else {
    auto& data_1 = *ctx->Input<Tensor>(1);
    ONNXRUNTIME_ENFORCE(data_1.Shape() == shape, "All inputs must have the same shape");

    mean = EigenMap<float>(data_0) + EigenMap<float>(data_1);
    for (int index = 2; index < inputCount; index++) {
      auto& data_n = *ctx->Input<Tensor>(index);
      ONNXRUNTIME_ENFORCE(data_n.Shape() == shape, "All inputs must have the same shape");
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
  ONNXRUNTIME_ENFORCE(tensor_shape.Shape().GetDims().size() == 1, "Shape must be 1 dimensional as it's tensor data is a shape");

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

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Expand,
    8,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Expand_8<float>);

template <>
Status Scale<float>::Compute(OpKernelContext* ctx) const {
  auto& X = *ctx->Input<Tensor>(0);
  auto& Y = *ctx->Output(0, X.Shape());
  EigenMap<float>(Y) = scale_ * EigenMap<float>(X);
  return Status::OK();
}

}  // namespace onnxruntime
