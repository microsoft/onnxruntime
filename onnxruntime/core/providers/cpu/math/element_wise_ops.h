// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/cpu/element_wise_ranged_transform.h"

namespace onnxruntime {
namespace functors {

template<typename T>
struct Log final : public ElementWiseRangedTransform<T> {
  Status Init(const onnxruntime::NodeAttributes) {
    return Status::OK();
  }

  ElementWiseRangedTransform<T>* Copy() const {
    using T1 = typename std::remove_pointer<decltype(this)>::type;
    using T2 = typename std::remove_const<T1>::type;
    return new T2(*this);
  }
  
  float Cost() const final { return 15.0f; }

  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first; 
        ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = xm.log();
  }
};

template <typename T>
struct Abs final : public ElementWiseRangedTransform<T> {
  Status Init(const onnxruntime::NodeAttributes) {
    return Status::OK();
  }

  ElementWiseRangedTransform<T>* Copy() const {
    using T1 = typename std::remove_pointer<decltype(this)>::type;
    using T2 = typename std::remove_const<T1>::type;
    return new T2(*this);
  }

  float Cost() const final { return 1.0f; }

  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = xm.cwiseAbs();
  }
};

template <typename T>
struct Neg final : public ElementWiseRangedTransform<T> {
  Status Init(const onnxruntime::NodeAttributes) {
    return Status::OK();
  }

  ElementWiseRangedTransform<T>* Copy() const {
    using T1 = typename std::remove_pointer<decltype(this)>::type;
    using T2 = typename std::remove_const<T1>::type;
    return new T2(*this);
  }

  float Cost() const final { return 1.0f; }

  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = -xm;
  }
};

template <typename T>
struct Floor final : public ElementWiseRangedTransform<T> {
  Status Init(const onnxruntime::NodeAttributes) {
    return Status::OK();
  }

  ElementWiseRangedTransform<T>* Copy() const {
    using T1 = typename std::remove_pointer<decltype(this)>::type;
    using T2 = typename std::remove_const<T1>::type;
    return new T2(*this);
  }

  float Cost() const final { return 1.0f; }

  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = xm.floor();
  }
};

template <typename T>
struct Ceil final : public ElementWiseRangedTransform<T> {
  Status Init(const onnxruntime::NodeAttributes) {
    return Status::OK();
  }

  ElementWiseRangedTransform<T>* Copy() const {
    using T1 = typename std::remove_pointer<decltype(this)>::type;
    using T2 = typename std::remove_const<T1>::type;
    return new T2(*this);
  }

  float Cost() const final { return 1.0f; }

  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = xm.ceil();
  }
};

template <typename T>
struct Reciprocal final : public ElementWiseRangedTransform<T> {
  Status Init(const onnxruntime::NodeAttributes) {
    return Status::OK();
  }

  ElementWiseRangedTransform<T>* Copy() const {
    using T1 = typename std::remove_pointer<decltype(this)>::type;
    using T2 = typename std::remove_const<T1>::type;
    return new T2(*this);
  }

  float Cost() const final { return 1.0f; }

  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = xm.cwiseInverse();
  }
};

template <typename T>
struct Sqrt final : public ElementWiseRangedTransform<T> {
  Status Init(const onnxruntime::NodeAttributes) {
    return Status::OK();
  }

  ElementWiseRangedTransform<T>* Copy() const {
    using T1 = typename std::remove_pointer<decltype(this)>::type;
    using T2 = typename std::remove_const<T1>::type;
    return new T2(*this);
  }

  float Cost() const final { return 2.0f; }

  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = xm.cwiseSqrt();
  }
};

template <typename T>
struct Exp final : public ElementWiseRangedTransform<T> {
  Status Init(const onnxruntime::NodeAttributes) {
    return Status::OK();
  }

  ElementWiseRangedTransform<T>* Copy() const {
    using T1 = typename std::remove_pointer<decltype(this)>::type;
    using T2 = typename std::remove_const<T1>::type;
    return new T2(*this);
  }

  float Cost() const final { return 2.0f; }

  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = xm.exp();
  }
};
}

DEFINE_ELE_KERNEL(Log)
DEFINE_ELE_KERNEL(Abs)
DEFINE_ELE_KERNEL(Neg)
DEFINE_ELE_KERNEL(Floor)
DEFINE_ELE_KERNEL(Ceil)
DEFINE_ELE_KERNEL(Reciprocal)
DEFINE_ELE_KERNEL(Sqrt)
DEFINE_ELE_KERNEL(Exp)


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


class Pow final : public OpKernel {
 public:
  Pow(const OpKernelInfo& info) : OpKernel(info) {
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

// Max versions 8 - 12
// Version 8 added broadcast
// Version 12 added types support
class Min_8 final : public OpKernel {
 public:
  Min_8(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  template <typename T>
  struct ComputeImpl;
};

template <typename T>
class Max_6 final : public OpKernel {
 public:
  Max_6(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

// Max versions 8 - 12
// Version 8 added broadcast
// Version 12 added types support
class Max_8 final : public OpKernel {
 public:
  Max_8(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;

private:

 template<typename T>
 struct ComputeImpl;
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
class BitShift final : public OpKernel {
 public:
  explicit BitShift(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  bool shift_left_;
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
class Erf final : public OpKernel {
 public:
  Erf(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
auto MakeEigenArrayMap(Tensor& t) -> EigenVectorArrayMap<T> { return EigenVectorArrayMap<T>(t.template MutableData<T>(), t.Shape().Size()); }
template <typename T>
auto MakeEigenArrayMap(const Tensor& t) -> ConstEigenVectorArrayMap<T> { return ConstEigenVectorArrayMap<T>(t.template Data<T>(), t.Shape().Size()); }

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
    } else if (counters_[0] > counts_[0]) { // Keep original logic above so that in most case it is faster
      delta = counters_[0] / counts_[0];
      counters_[0] = counters_[0] % counts_[0];
      for (size_t counterIndex = 1; counterIndex < counters_.size(); counterIndex++) {
        index_ += delta * deltas_[counterIndex];
        counters_[counterIndex] += delta;
        if (counters_[counterIndex] < counts_[counterIndex]) break;
        delta = counters_[counterIndex] / counts_[counterIndex];
        counters_[counterIndex] = counters_[counterIndex] % counts_[counterIndex];
      }
    }
    return index;
  }

  void Reserve(int64_t max_dims) {
    deltas_.reserve(static_cast<size_t>(max_dims));
    counts_.reserve(static_cast<size_t>(max_dims));
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
  ptrdiff_t count_{1};  // Running total count of entries in tensor, used while building up the entries

 private:
  size_t index_{};
};

struct Broadcaster {
  Broadcaster(const std::vector<int64_t>& shape1, const std::vector<int64_t>& shape2) {
    size_t dimension_count_max = std::max(shape1.size(), shape2.size());
    size_t dimension_count_min = std::min(shape1.size(), shape2.size());
    output_shape_.resize(dimension_count_max);
    iterator1_.Reserve(dimension_count_max);
    iterator2_.Reserve(dimension_count_max);

    auto iter1 = shape1.end();
    auto iter2 = shape2.end();
    auto output_shape = output_shape_.end();

    // Scalars are a special case, as it's always a broadcast
    size_t index = 0;
    if (dimension_count_min == 0) {
      if (shape1.empty())  // Shape1 is a scalar
      {
        if (shape2.empty())  // Two scalars?
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
    } else {
      for (; index < dimension_count_min; index++) {
        auto axis1 = *--iter1;
        auto axis2 = *--iter2;

        auto largest = std::max(axis1, axis2);
        auto smallest = std::min(axis1, axis2);
        auto dim_to_use = largest;

        if (smallest == 0) {
          ORT_ENFORCE(largest <= 1, "Can broadcast 0 by 0 or 1. ", largest, " is invalid.");
          dim_to_use = smallest;
        }

        *--output_shape = dim_to_use;

        // if both 1, or a 1 and 0, and there are more dims, we can let the next iteration do the Init
        if (dim_to_use <= 1 && index + 1 < dimension_count_min)
          continue;

        iterator1_.Init(axis1, dim_to_use);
        iterator2_.Init(axis2, dim_to_use);
        index++;  // Manually increment since we processed one axis
        break;
      }
    }

    for (; index < dimension_count_min; index++) {
      auto axis1 = *--iter1;
      auto axis2 = *--iter2;

      auto largest = std::max(axis1, axis2);
      auto smallest = std::min(axis1, axis2);
      auto dim_to_use = largest;

      if (smallest == 0) {
        ORT_ENFORCE(largest <= 1, "Can broadcast 0 by 0 or 1. ", largest, " is invalid.");
        dim_to_use = smallest;
      }

      *--output_shape = dim_to_use;

      if (largest == 1)  // Nothing to do in this case
        continue;

      iterator1_.Append(axis1, dim_to_use);
      iterator2_.Append(axis2, dim_to_use);
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

  void AdvanceBy(size_t offset) {
    ORT_ENFORCE(offset % span_size_ == 0, "TBroadcaster can only start at span boundary!");
    broadcaster_.iterator1_.AdvanceBy(offset);
    broadcaster_.iterator2_.AdvanceBy(offset);
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
  TBroadcastOutput(size_t span_size, Tensor& tensor, int64_t start_offset = 0, int64_t end_offset = 0)
      : span_size_(span_size) {
    int64_t len = tensor.Shape().Size();
    int64_t real_end = (end_offset <= 0) ? len : end_offset;
    if (start_offset != 0 || end_offset != 0) { // Keep original semantic
      ORT_ENFORCE(start_offset >= 0 && real_end >= 0 && start_offset <= real_end && real_end <= len,
                  "Invalid start/ending offset [", start_offset, ",", real_end, ") for tensor of length:", len);
      ORT_ENFORCE(start_offset % span_size == 0 && real_end % span_size == 0,
                  "Broadcast Output range [", start_offset, ", ", real_end,
                  ") are not at boundary of span with size:", span_size);
    }
    output_ = tensor.template MutableData<T>() + start_offset;
    output_end_ = tensor.template MutableData<T>() + real_end;
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
    auto status = context.GetTempSpaceAllocator(&allocator_);
    ORT_ENFORCE(status.IsOK(), status.ErrorMessage());
  }

  std::unique_ptr<Tensor> Allocate(const TensorShape& shape) {
    return onnxruntime::make_unique<Tensor>(DataTypeImpl::GetType<T>(),
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
void BroadcastOneSpan(concurrency::ThreadPool* tp, double unit_cost, TOutput* output_ptr, int64_t output_size, 
                    const TInput* input0_ptr, int64_t input0_size, const TInput* input1_ptr, int64_t input1_size,
                             Input0Scalar input0scalar, Input1Scalar input1scalar, General general) {
  if (input0_size == 1) {
    ORT_ENFORCE(input1_size == output_size);
    concurrency::ThreadPool::TryParallelFor(tp, output_size, 
                                {static_cast<float>(sizeof(TInput)), static_cast<float>(sizeof(TOutput)), unit_cost},
                               [=](std::ptrdiff_t first, std::ptrdiff_t last) {
                                 size_t count = static_cast<size_t>(last - first);
                                 EigenVectorMap<TOutput> output_map(output_ptr + first, count);
                                 ConstEigenVectorMap<TInput> input1_map(input1_ptr + first, count); 
                                 input0scalar(output_map, *input0_ptr, input1_map);
                               });
  } else if (input1_size == 1) {
    ORT_ENFORCE(input0_size == output_size);
    concurrency::ThreadPool::TryParallelFor(tp, output_size,
                                {static_cast<float>(sizeof(TInput)), static_cast<float>(sizeof(TOutput)), unit_cost},
                               [=](std::ptrdiff_t first, std::ptrdiff_t last) {
                                 size_t count = static_cast<size_t>(last - first);
                                 EigenVectorMap<TOutput> output_map(output_ptr + first, count);
                                 ConstEigenVectorMap<TInput> input0_map(input0_ptr + first, count);
                                 input1scalar(output_map, input0_map, *input1_ptr);
                               });
  } else {
    concurrency::ThreadPool::TryParallelFor(tp, output_size,
                                {static_cast<float>(sizeof(TInput)), static_cast<float>(sizeof(TOutput)), unit_cost},
                               [=](std::ptrdiff_t first, std::ptrdiff_t last) {
                                 size_t count = static_cast<size_t>(last - first);
                                 EigenVectorMap<TOutput> output_map(output_ptr + first, count);
                                 ConstEigenVectorMap<TInput> input0_map(input0_ptr + first, count);
                                 ConstEigenVectorMap<TInput> input1_map(input1_ptr + first, count); 
                                 general(output_map, input0_map, input1_map);
                               });
  }
}

template <typename TInput, typename TOutput, typename Input0Scalar, typename Input1Scalar, typename General>
Status BroadcastTwo(OpKernelContext& context, Input0Scalar input0scalar, Input1Scalar input1scalar, General general, double unit_cost=-1.0f) {
  if (unit_cost == -1.0f) { // no paralellization 
    TBroadcaster<TInput, TInput> bc(*context.Input<Tensor>(0), *context.Input<Tensor>(1));
    TBroadcastOutput<TOutput> output(bc.GetSpanSize(), *context.Output(0, bc.GetOutputShape()));
    BroadcastLoop(bc, output, input0scalar, input1scalar, general);
  } else {
    const Tensor* input0_tensor = context.Input<Tensor>(0);
    const Tensor* input1_tensor = context.Input<Tensor>(1);
    TBroadcaster<TInput, TInput> bc(*input0_tensor, *input1_tensor);
    Tensor& output_tensor = *context.Output(0, bc.GetOutputShape());
    auto span_size = bc.GetSpanSize();
    int64_t output_size = output_tensor.Shape().Size();
    if (output_size != 0) {
      concurrency::ThreadPool* tp = context.GetOperatorThreadPool();
      if (span_size != 0) {
        if (output_size == static_cast<int64_t>(span_size)) {  // Only one big span for all data, parallel inside it
          ORT_ENFORCE((output_size % span_size) == 0);
          BroadcastOneSpan(tp, unit_cost, output_tensor.MutableData<TOutput>(), output_size,
                           input0_tensor->Data<TInput>(), input0_tensor->Shape().Size(),
                           input1_tensor->Data<TInput>(), input1_tensor->Shape().Size(),
                           input0scalar, input1scalar, general);
        } else {
          concurrency::ThreadPool::TryParallelFor(
              tp, output_size / span_size,
              {static_cast<float>(sizeof(TInput)) * span_size, static_cast<float>(sizeof(TOutput)) * span_size, unit_cost * span_size},
              [=, &bc, &output_tensor](std::ptrdiff_t first_span, std::ptrdiff_t last_span) {
                TBroadcaster<TInput, TInput> span_bc(bc);
                TBroadcastOutput<TOutput> span_output(span_size, output_tensor, first_span * span_size, last_span * span_size);
                span_bc.AdvanceBy(first_span * span_size);
                BroadcastLoop(span_bc, span_output, input0scalar, input1scalar, general);
              });
        }
      }
    }
  }
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
