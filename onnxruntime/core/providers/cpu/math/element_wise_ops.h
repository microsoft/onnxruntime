// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {

template <typename T>
class Add final : public OpKernel {
 public:
  Add(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Sub final : public OpKernel {
 public:
  Sub(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Mul final : public OpKernel {
 public:
  Mul(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Div final : public OpKernel {
 public:
  Div(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Abs final : public OpKernel {
 public:
  Abs(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    auto& input = *context->Input<Tensor>(0);
    auto& output = *context->Output(0, input.Shape());

    EigenMap<T>(output) = EigenMap<T>(input).cwiseAbs();
    return Status::OK();
  }
};

template <typename T>
class Neg final : public OpKernel {
 public:
  Neg(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    auto& input = *context->Input<Tensor>(0);
    auto& output = *context->Output(0, input.Shape());

    EigenMap<T>(output) = -EigenMap<T>(input);
    return Status::OK();
  }
};

template <typename T>
class Floor final : public OpKernel {
 public:
  Floor(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Ceil final : public OpKernel {
 public:
  Ceil(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Reciprocal final : public OpKernel {
 public:
  Reciprocal(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Sqrt final : public OpKernel {
 public:
  Sqrt(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Pow final : public OpKernel {
 public:
  Pow(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Exp final : public OpKernel {
 public:
  Exp(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Log final : public OpKernel {
 public:
  Log(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Sum_6 : public OpKernel {
 public:
  Sum_6(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Sum_8 final : public OpKernel {
 public:
  Sum_8(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Min_6 final : public OpKernel {
 public:
  Min_6(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Min_8 final : public OpKernel {
 public:
  Min_8(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Max_6 final : public OpKernel {
 public:
  Max_6(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Max_8 final : public OpKernel {
 public:
  Max_8(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

class Not final : public OpKernel {
 public:
  Not(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

class And final : public OpKernel {
 public:
  And(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

class Or final : public OpKernel {
 public:
  Or(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

class Xor final : public OpKernel {
 public:
  Xor(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Equal final : public OpKernel {
 public:
  Equal(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Less final : public OpKernel {
 public:
  Less(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Greater final : public OpKernel {
 public:
  Greater(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Mean_6 final : public OpKernel {
 public:
  Mean_6(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Mean_8 final : public OpKernel {
 public:
  Mean_8(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Affine final : public OpKernel {
 public:
  Affine(const OpKernelInfo& info) : OpKernel(info) {
    // Either model-supplied or default values should be returned for alpha and beta
    ORT_ENFORCE(info.GetAttr("alpha", &alpha_).IsOK());
    ORT_ENFORCE(info.GetAttr("beta", &beta_).IsOK());
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  float alpha_;
  float beta_;
};

// PRelu is activation function, but it's closer to binary elementwise ops in implementation
template <typename T>
class PRelu final : public OpKernel {
 public:
  PRelu(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Expand_8 final : public OpKernel {
 public:
  Expand_8(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Scale final : public OpKernel {
 public:
  Scale(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttr("scale", &scale_).IsOK());
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  float scale_;
};

template <typename T>
class Erf final : public OpKernel {
 public:
  Erf(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

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
    ORT_ENFORCE(axis == 1 || axis == largest, "Attempting to broadcast an axis by a dimension other than 1. ", axis, " by ", largest);

    deltas_.push_back(axis > 1);
    counts_.push_back(largest);
    count_ *= axis;
  }

  void Append(int64_t axis, int64_t largest) {
    ORT_ENFORCE(axis == 1 || axis == largest, "Attempting to broadcast an axis by a dimension other than 1. ", axis, " by ", largest);

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

template <typename T0, typename T1>
struct TBroadcaster {
  TBroadcaster(const Tensor& input0, const Tensor& input1)
      : input_tensor0_(input0),
        input_tensor1_(input1) {
  }

  TensorShape GetOutputShape() const { return TensorShape(broadcaster_.output_shape_); }
  size_t GetSpanSize() const { return span_size_; }

  bool IsInput0Scalar() const { return broadcaster_.iterator1_.deltas_.front() == 0; }
  bool IsInput1Scalar() const { return broadcaster_.iterator2_.deltas_.front() == 0; }

  const T0& NextScalar0() { return *Next0(); }
  const T1& NextScalar1() { return *Next1(); }

  gsl::span<const T0> NextSpan0() { return gsl::span<const T0>(Next0(), span_size_); }
  gsl::span<const T1> NextSpan1() { return gsl::span<const T1>(Next1(), span_size_); }

  ConstEigenVectorMap<T0> NextEigen0() { return ConstEigenVectorMap<T0>(Next0(), span_size_); }
  ConstEigenVectorMap<T1> NextEigen1() { return ConstEigenVectorMap<T1>(Next1(), span_size_); }

 private:
  const T0* Next0() { return input0_ + broadcaster_.iterator1_.AdvanceBy(span_size_); }
  const T1* Next1() { return input1_ + broadcaster_.iterator2_.AdvanceBy(span_size_); }

  const Tensor& input_tensor0_;
  const Tensor& input_tensor1_;
  Broadcaster broadcaster_{input_tensor0_.Shape().GetDims(), input_tensor1_.Shape().GetDims()};
  size_t span_size_{broadcaster_.GetSpanSize()};

  const T0* input0_{input_tensor0_.template Data<T0>()};
  const T1* input1_{input_tensor1_.template Data<T1>()};
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
    ORT_ENFORCE(context.GetTempSpaceAllocator(&allocator_).IsOK());
  }

  std::unique_ptr<Tensor> Allocate(const TensorShape& shape) {
    return std::make_unique<Tensor>(DataTypeImpl::GetType<T>(),
                                    shape,
                                    allocator_);
  }

 private:
  AllocatorPtr allocator_;
};

// Broadcast loop for when using eigen, functions are in this form:
// Input0Scalar: [](EigenVectorMap<TOutput> output, TInput0 input0, ConstEigenVectorMap<TInput1> input1)
// Input1Scalar: [](EigenVectorMap<TOutput> output, ConstEigenVectorMap<TInput0> input0, TInput1 input1)
// General     : [](EigenVectorMap<TOutput> output, ConstEigenVectorMap<TInput0> input0,
//                  ConstEigenVectorMap<TInput1> input1)
// Scalar parameters can also be of type const TX&.
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

// Broadcast loop for when using gsl::span<T>, functions are in this form:
// Input0Scalar: [](gsl::span<TOutput> output, TInput0 input0, gsl::span<const TInput1> input1)
// Input1Scalar: [](gsl::span<TOutput> output, gsl::span<const TInput0> input0, TInput1 input1)
// General     : [](gsl::span<TOutput> output, gsl::span<const TInput0> input0, gsl::span<const TInput1> input1)
// Scalar parameters can also be of type const TX&.
template <typename TBroadcaster, typename Output, typename Input0Scalar, typename Input1Scalar, typename General>
void BroadcastLoopSpan(TBroadcaster& bc, Output& output, Input0Scalar input0scalar, Input1Scalar input1scalar, General general) {
  if (bc.IsInput0Scalar()) {
    while (output)
      input0scalar(output.NextSpanOutput(), bc.NextScalar0(), bc.NextSpan1());
  } else if (bc.IsInput1Scalar()) {
    while (output)
      input1scalar(output.NextSpanOutput(), bc.NextSpan0(), bc.NextScalar1());
  } else {
    while (output)
      general(output.NextSpanOutput(), bc.NextSpan0(), bc.NextSpan1());
  }
}

template <typename TInput, typename TOutput, typename Input0Scalar, typename Input1Scalar, typename General>
Status BroadcastTwo(OpKernelContext& context, Input0Scalar input0scalar, Input1Scalar input1scalar, General general) {
  TBroadcaster<TInput, TInput> bc(*context.Input<Tensor>(0), *context.Input<Tensor>(1));
  TBroadcastOutput<TOutput> output(bc.GetSpanSize(), *context.Output(0, bc.GetOutputShape()));
  BroadcastLoop(bc, output, input0scalar, input1scalar, general);

  return Status::OK();
}

template <typename TInput, typename TOutput, typename Input0Scalar, typename Input1Scalar, typename General>
Status BroadcastVariadic(const Node& node, OpKernelContext& context, Input0Scalar input0scalar, Input1Scalar input1scalar, General general) {
  auto input_count = node.InputArgCount().front();
  ORT_ENFORCE(input_count >= 1, "Must have 1 or more inputs");

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

    TBroadcaster<TInput, TInput> bc(tensor0, tensor1);

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

}  // namespace onnxruntime
