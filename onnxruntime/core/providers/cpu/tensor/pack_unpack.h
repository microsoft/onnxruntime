// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <limits>

#include "core/common/safeint.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/tensor/pack_unpack_impl.h"

namespace onnxruntime {
namespace packing {

struct Attributes {
  Parameters parameters{};
  int64_t axis;
  int64_t output_type;

  template <typename Info>
  explicit Attributes(const Info& info) {
    int64_t bits = 0, radix = 0, group_size = 0;
    const bool has_bits = info.GetAttr("bits", &bits).IsOK();
    const bool has_radix = info.GetAttr("radix", &radix).IsOK();
    const bool has_group_size = info.GetAttr("group_size", &group_size).IsOK();
    ORT_ENFORCE((has_bits && !has_radix && !has_group_size) || (!has_bits && has_radix && has_group_size),
                "Pack/Unpack: specify either bits or both radix and group_size, not both.");
    if (has_bits) {
      ORT_ENFORCE(bits >= 1 && bits <= 8, "Pack/Unpack: bits must be in [1, 8]; received ", bits, ".");
      radix = int64_t{1} << bits;
      group_size = 1;
    }
    ORT_ENFORCE(radix >= 2 && radix <= 256 && group_size >= 1,
                "Pack/Unpack: radix must be in [2, 256] and group_size must be positive; received ",
                radix, " and ", group_size, ".");
    uint64_t limit = 1;
    for (int64_t j = 0; j < group_size; ++j) {
      ORT_ENFORCE(limit <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / radix,
                  "Pack/Unpack: radix^group_size exceeds positive INT64; reduce radix or group_size.");
      limit *= radix;
    }
    int width = 0;
    for (uint64_t maximum = limit - 1; maximum != 0; maximum >>= 1) ++width;
    parameters = {static_cast<uint64_t>(radix), static_cast<uint64_t>(group_size), width};
    axis = info.template GetAttrOrDefault<int64_t>("axis", -1);
    output_type = info.template GetAttrOrDefault<int64_t>("output_type", ONNX_NAMESPACE::TensorProto::UINT8);
    ORT_ENFORCE(output_type == ONNX_NAMESPACE::TensorProto::UINT8 ||
                    output_type == ONNX_NAMESPACE::TensorProto::UINT32,
                "Pack: output_type must be UINT8 (2) or UINT32 (12); received ", output_type, ".");
  }

  int64_t WordCount(int64_t count, int word_bits) const {
    const int64_t group_size = static_cast<int64_t>(parameters.group_size);
    const int64_t groups = count / group_size + (count % group_size != 0);
    // Do not form groups * width: it can overflow even when the word count fits.
    const int64_t maximum = std::numeric_limits<int64_t>::max();
    ORT_ENFORCE(groups / word_bits <= maximum / parameters.width,
                "Pack/Unpack: packed axis dimension exceeds INT64; reduce count or group_size.");
    const int64_t whole = (groups / word_bits) * parameters.width;
    const int64_t tail = ((groups % word_bits) * parameters.width + word_bits - 1) / word_bits;
    ORT_ENFORCE(whole <= maximum - tail,
                "Pack/Unpack: packed axis dimension exceeds INT64; reduce count or group_size.");
    return whole + tail;
  }

  Status Prepare(const Tensor& input, const Tensor* count_input, bool unpack,
                 TensorShapeVector& output_shape, Layout& layout) const {
    const auto& shape = input.Shape();
    const int64_t rank = static_cast<int64_t>(shape.NumDimensions());
    ORT_RETURN_IF_NOT(rank > 0 && axis >= -rank && axis < rank,
                      "Pack/Unpack: input must have rank at least 1 and axis must be in [-rank, rank-1]; rank=",
                      rank, ", axis=", axis, ".");
    const size_t normalized_axis = static_cast<size_t>(axis < 0 ? axis + rank : axis);
    const int type = unpack ? input.GetElementType() : static_cast<int>(output_type);
    ORT_RETURN_IF_NOT(type == ONNX_NAMESPACE::TensorProto::UINT8 || type == ONNX_NAMESPACE::TensorProto::UINT32,
                      "Unpack: storage input must be UINT8 or UINT32.");
    ORT_RETURN_IF_NOT(unpack || input.GetElementType() == ONNX_NAMESPACE::TensorProto::UINT8,
                      "Pack: logical input codes must be UINT8.");
    int64_t count = shape[normalized_axis];
    if (unpack) {
      ORT_RETURN_IF_NOT(count_input && count_input->GetElementType() == ONNX_NAMESPACE::TensorProto::INT64 &&
                            count_input->Shape().NumDimensions() == 0,
                        "Unpack: count must be a nonnegative scalar INT64 tensor.");
      count = *count_input->Data<int64_t>();
      ORT_RETURN_IF_NOT(count >= 0, "Unpack: count must be nonnegative; received ", count, ".");
    }
    const int64_t words = WordCount(count, type == ONNX_NAMESPACE::TensorProto::UINT32 ? 32 : 8);
    ORT_RETURN_IF_NOT(!unpack || shape[normalized_axis] == words,
                      "Unpack: packed axis dimension must equal the required word count ", words,
                      " for count=", count, "; received ", shape[normalized_axis],
                      ". Adjust count or supply the exact packed shape.");
    output_shape.assign(shape.GetDims().begin(), shape.GetDims().end());
    output_shape[normalized_axis] = unpack ? count : words;
    layout = {static_cast<uint64_t>(count), static_cast<uint64_t>(words),
              static_cast<uint64_t>(count / parameters.group_size + (count % parameters.group_size != 0)), 0};
    // Empty tensors preserve the shape rule without multiplying huge unused dimensions.
    if (shape.Size() == 0 || count == 0) return Status::OK();
    layout.inner = static_cast<uint64_t>(shape.SizeFromDimension(normalized_axis + 1));
    return Status::OK();
  }
};

}  // namespace packing
}  // namespace onnxruntime
