// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cstdint>
#include <limits>

#include "core/common/safeint.h"
#include "core/framework/op_kernel.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace {

class PackingBase : public OpKernel {
 public:
  explicit PackingBase(const OpKernelInfo& info) : OpKernel(info) {
    axis_ = info.GetAttrOrDefault<int64_t>("axis", -1);
    int64_t bits = 0;
    int64_t radix = 0;
    int64_t group_size = 0;
    const bool has_bits = info.GetAttr("bits", &bits).IsOK();
    const bool has_radix = info.GetAttr("radix", &radix).IsOK();
    const bool has_group_size = info.GetAttr("group_size", &group_size).IsOK();
    ORT_ENFORCE((has_bits && !has_radix && !has_group_size) ||
                    (!has_bits && has_radix && has_group_size),
                "Specify either bits or both radix and group_size.");
    if (has_bits) {
      ORT_ENFORCE(bits >= 1 && bits <= 8, "bits must be in [1, 8].");
      radix = int64_t{1} << bits;
      group_size = 1;
    }
    ORT_ENFORCE(radix >= 2 && radix <= 256, "radix must be in [2, 256].");
    ORT_ENFORCE(group_size > 0, "group_size must be positive.");
    radix_ = static_cast<uint64_t>(radix);
    group_size_ = group_size;
    radix_power_ = 1;
    for (int64_t i = 0; i < group_size_; ++i) {
      ORT_ENFORCE(radix_power_ <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / radix_,
                  "radix^group_size must fit in positive INT64.");
      radix_power_ *= radix_;
    }
    for (uint64_t value = radix_power_ - 1; value != 0; value >>= 1) {
      ++group_width_;
    }
  }

 protected:
  int64_t PackedLength(int64_t count, int word_bits) const {
    const int64_t groups = count / group_size_ + (count % group_size_ != 0);
    // Divide before multiplying so an otherwise representable word count does not overflow.
    return SafeInt<int64_t>(groups / word_bits) * group_width_ +
           (groups % word_bits * group_width_ + word_bits - 1) / word_bits;
  }

  int64_t axis_;
  uint64_t radix_;
  int64_t group_size_;
  uint64_t radix_power_;
  int group_width_ = 0;
};

class Pack final : public PackingBase {
 public:
  explicit Pack(const OpKernelInfo& info) : PackingBase(info) {
    const auto type = info.GetAttrOrDefault<int64_t>("output_type", ONNX_NAMESPACE::TensorProto::UINT8);
    ORT_ENFORCE(type == ONNX_NAMESPACE::TensorProto::UINT8 || type == ONNX_NAMESPACE::TensorProto::UINT32,
                "output_type must be UINT8 or UINT32.");
    uint32_output_ = type == ONNX_NAMESPACE::TensorProto::UINT32;
  }

  Status Compute(OpKernelContext* context) const override {
    const auto& input = *context->Input<Tensor>(0);
    const auto& shape = input.Shape();
    ORT_RETURN_IF_NOT(shape.NumDimensions() >= 1, "Pack requires an input of rank at least 1.");
    const auto axis = static_cast<size_t>(HandleNegativeAxis(axis_, shape.NumDimensions()));
    TensorShapeVector output_dims(shape.GetDims().begin(), shape.GetDims().end());
    output_dims[axis] = PackedLength(shape[axis], uint32_output_ ? 32 : 8);
    auto& output = *context->Output(0, output_dims);
    if (output.Shape().Size() == 0) {
      return Status::OK();
    }
    return uint32_output_ ? ComputeTyped<uint32_t>(input, output, axis)
                          : ComputeTyped<uint8_t>(input, output, axis);
  }

 private:
  bool uint32_output_;

  template <typename Word>
  Status ComputeTyped(const Tensor& input, Tensor& output, size_t axis) const {
    constexpr int word_bits = sizeof(Word) * 8;
    const auto* src = input.Data<uint8_t>();
    auto* dst = output.MutableData<Word>();
    std::fill_n(dst, output.Shape().Size(), Word{0});
    const int64_t count = input.Shape()[axis];
    const int64_t words = output.Shape()[axis];
    const int64_t outer = input.Shape().SizeToDimension(axis);
    const int64_t inner = input.Shape().SizeFromDimension(axis + 1);
    for (int64_t row = 0; row < outer; ++row) {
      for (int64_t lane = 0; lane < inner; ++lane) {
        int64_t word = 0;
        int offset = 0;
        for (int64_t start = 0; start < count;) {
          const int64_t digits = std::min(group_size_, count - start);
          uint64_t index = 0;
          for (int64_t digit = digits; digit > 0; --digit) {
            const auto code = src[(row * count + start + digit - 1) * inner + lane];
            ORT_RETURN_IF_NOT(code < radix_, "Pack input code must be less than radix.");
            index = index * radix_ + code;
          }
          for (int remaining = group_width_; remaining > 0;) {
            const int chunk = std::min(remaining, word_bits - offset);
            const uint64_t mask = (uint64_t{1} << chunk) - 1;
            dst[(row * words + word) * inner + lane] |= static_cast<Word>((index & mask) << offset);
            index >>= chunk;
            remaining -= chunk;
            offset += chunk;
            if (offset == word_bits) {
              offset = 0;
              ++word;
            }
          }
          start += digits;
        }
      }
    }
    return Status::OK();
  }
};

class Unpack final : public PackingBase {
 public:
  explicit Unpack(const OpKernelInfo& info) : PackingBase(info) {}

  Status Compute(OpKernelContext* context) const override {
    const auto& input = *context->Input<Tensor>(0);
    const auto& shape = input.Shape();
    ORT_RETURN_IF_NOT(shape.NumDimensions() >= 1, "Unpack requires an input of rank at least 1.");
    const auto axis = static_cast<size_t>(HandleNegativeAxis(axis_, shape.NumDimensions()));
    const auto& count_tensor = *context->Input<Tensor>(1);
    ORT_RETURN_IF_NOT(count_tensor.Shape().NumDimensions() == 0, "count must be a scalar INT64 tensor.");
    const int64_t count = *count_tensor.Data<int64_t>();
    ORT_RETURN_IF_NOT(count >= 0, "count must be nonnegative.");
    const bool uint32_input = input.IsDataType<uint32_t>();
    ORT_RETURN_IF_NOT(shape[axis] == PackedLength(count, uint32_input ? 32 : 8),
                      "Unpack input axis dimension must equal the packed word count.");
    TensorShapeVector output_dims(shape.GetDims().begin(), shape.GetDims().end());
    output_dims[axis] = count;
    auto& output = *context->Output(0, output_dims);
    if (output.Shape().Size() == 0) {
      return Status::OK();
    }
    return uint32_input ? ComputeTyped<uint32_t>(input, output, axis)
                        : ComputeTyped<uint8_t>(input, output, axis);
  }

 private:
  template <typename Word>
  Status ComputeTyped(const Tensor& input, Tensor& output, size_t axis) const {
    constexpr int word_bits = sizeof(Word) * 8;
    const auto* src = input.Data<Word>();
    auto* dst = output.MutableData<uint8_t>();
    const int64_t count = output.Shape()[axis];
    const int64_t words = input.Shape()[axis];
    const int64_t outer = input.Shape().SizeToDimension(axis);
    const int64_t inner = input.Shape().SizeFromDimension(axis + 1);
    for (int64_t row = 0; row < outer; ++row) {
      for (int64_t lane = 0; lane < inner; ++lane) {
        int64_t word = 0;
        int offset = 0;
        for (int64_t start = 0; start < count;) {
          uint64_t index = 0;
          int consumed = 0;
          for (int remaining = group_width_; remaining > 0;) {
            const int chunk = std::min(remaining, word_bits - offset);
            const uint64_t mask = (uint64_t{1} << chunk) - 1;
            const uint64_t value = src[(row * words + word) * inner + lane];
            index |= ((value >> offset) & mask) << consumed;
            consumed += chunk;
            remaining -= chunk;
            offset += chunk;
            if (offset == word_bits) {
              offset = 0;
              ++word;
            }
          }
          ORT_RETURN_IF_NOT(index < radix_power_, "Unpack group index must be less than radix^group_size.");
          const int64_t digits = std::min(group_size_, count - start);
          for (int64_t digit = 0; digit < digits; ++digit) {
            dst[(row * count + start + digit) * inner + lane] = static_cast<uint8_t>(index % radix_);
            index /= radix_;
          }
          start += digits;
        }
      }
    }
    return Status::OK();
  }
};

}  // namespace

ONNX_CPU_OPERATOR_KERNEL(
    Pack, 29,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("TStorage", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<uint32_t>()}),
    Pack);

ONNX_CPU_OPERATOR_KERNEL(
    Unpack, 29,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("TStorage", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<uint32_t>()}),
    Unpack);

}  // namespace onnxruntime
