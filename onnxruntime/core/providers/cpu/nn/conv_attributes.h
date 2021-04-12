// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifndef SHARED_PROVIDER
#include "core/common/common.h"
#include "core/providers/common.h"
#include "core/util/math.h"
#endif
#include "core/framework/op_node_proto_helper.h"

namespace onnxruntime {

// A helper struct holding attributes for Conv-family ops
struct ConvAttributes {
  explicit ConvAttributes(const OpKernelInfo& info) {
    std::string auto_pad_str;
    auto status = info.GetAttr<std::string>("auto_pad", &auto_pad_str);
    if (status.IsOK()) {
      auto_pad = StringToAutoPadType(auto_pad_str);
    }

    kernel_shape_specified = info.GetAttrs<int64_t>("kernel_shape", kernel_shape_).IsOK();

    status = info.GetAttrs<int64_t>("strides", strides);
    if (!status.IsOK() || strides.empty()) {
      strides.resize(kernel_shape_.size(), 1);
    }

    status = info.GetAttrs<int64_t>("pads", pads);
    if (!status.IsOK()) {
      // If pads are not explicitly provided, fill the container with all zeros
      // so that we can compute and fill in pad values downstream
      pads.resize(kernel_shape_.size() * 2, 0);
    } else {
      // Pads are explicitly provided, make sure that auto_pad is NOTSET
      ORT_ENFORCE(auto_pad == AutoPadType::NOTSET,
                  "A Conv/ConvTranspose node has both 'auto_pad' and 'pads' attributes");
    }

    status = info.GetAttrs<int64_t>("dilations", dilations);
    if (!status.IsOK() || dilations.empty()) {
      dilations.resize(kernel_shape_.size(), 1);
    }

    status = info.GetAttr<int64_t>("group", &group);
    if (!status.IsOK()) {
      group = 1;
    }

#if false
    // TODO: Re-enable when attributes values are guaranteed to be filled.
    // Make sure empty strides or dilations are defaulted to 1 if necessary
    std::string auto_pad_str;
    ORT_ENFORCE(info.GetAttr<std::string>("auto_pad", &auto_pad_str).IsOK());
    auto_pad = StringToAutoPadType(auto_pad_str);
    ORT_ENFORCE(info.GetAttr<int64_t>("group", &group).IsOK());
    ORT_ENFORCE(info.GetAttrs<int64_t>("kernel_shape", kernel_shape_).IsOK());
    ORT_ENFORCE(info.GetAttrs<int64_t>("strides", strides).IsOK());
    ORT_ENFORCE(info.GetAttrs<int64_t>("pads", pads).IsOK());
    ORT_ENFORCE(info.GetAttrs<int64_t>("dilations", dilations).IsOK());
#endif
  }

  ~ConvAttributes() = default;

  Status ComputeKernelShape(const TensorShape& weight_shape, std::vector<int64_t>& kernel_shape) const {
    if (kernel_shape_specified) {
      kernel_shape = kernel_shape_;
      if (kernel_shape.size() + 2 != weight_shape.NumDimensions()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "kernel_shape num_dims is not compatible with W num_dims.",
                               " kernel_shape: ", TensorShape(kernel_shape).ToString().c_str(),
                               " W: ", weight_shape.ToString().c_str());
      }
      for (size_t i = 0; i < kernel_shape.size(); ++i) {
        if (kernel_shape[i] != weight_shape[i + 2]) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "kernel_shape is not compatible with W shape.",
                                 " kernel_shape: ", TensorShape(kernel_shape).ToString().c_str(),
                                 " W: ", weight_shape.ToString().c_str());
        }
      }
    } else {
      auto& weight_dims = weight_shape.GetDims();
      kernel_shape = std::vector<int64_t>(weight_dims.begin() + 2, weight_dims.end());
    }

    return Status::OK();
  }

  Status ValidateInputShape(const TensorShape& input_shape,
                            const TensorShape& weight_shape,
                            bool channels_last = false) const {
    if (input_shape.NumDimensions() != weight_shape.NumDimensions()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "X num_dims does not match W num_dims.",
                             " X: ", input_shape.ToString().c_str(),
                             " W: ", weight_shape.ToString().c_str());
    }

    const int64_t M = weight_shape[0];
    const int64_t C = channels_last ? input_shape.GetDims().back() : input_shape[1];

    if (C != weight_shape[1] * group) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Input channels C is not equal to kernel channels * group.",
                             " C: ", C,
                             " kernel channels: ", weight_shape[1],
                             " group: ", group);
    }

    if (M % group != 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Output channels M is not divisible by group.",
                             " M: ", M,
                             " group: ", group);
    }
    return Status::OK();
  }

  Status ValidateInputShape(const Tensor* input, const Tensor* weight) const {
    return ValidateInputShape(input->Shape(), weight->Shape());
  }

  Status InferOutputShape(const TensorShape& input_shape,
                          const std::vector<int64_t>& kernel_shape,
                          const std::vector<int64_t>& strides_p,
                          const std::vector<int64_t>& dilations_p,
                          std::vector<int64_t>& pads_p,
                          std::vector<int64_t>& output_shape,
                          bool force_symmetric_auto_padding = false) const {
    size_t rank = input_shape.NumDimensions();

    // Make sure all "metadata" containers have the right number of elements
    if (rank > strides_p.size())
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Not enough elements in strides. Expected: ", rank, " Got: ", strides_p.size());

    if (rank > kernel_shape.size())
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Not enough elements in kernel shape. Expected: ", rank, " Got: ", kernel_shape.size());

    if (rank > dilations_p.size())
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Not enough elements in dilations. Expected: ", rank, " Got: ", dilations_p.size());

    if ((2 * rank) > pads_p.size())
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Not enough elements in pads. Expected: ", (2 * rank), " Got: ", pads_p.size());

    for (size_t dim = 0; dim < rank; ++dim) {
      int64_t output_dim_size = 0;
      ORT_RETURN_IF_ERROR(ComputePadAndOutputShape(input_shape[dim],
                                                   strides_p[dim],
                                                   kernel_shape[dim],
                                                   dilations_p[dim],
                                                   auto_pad,
                                                   pads_p.at(dim),
                                                   pads_p.at(input_shape.NumDimensions() + dim),
                                                   output_dim_size,
                                                   force_symmetric_auto_padding));
      if (output_dim_size <= 0) {
        return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Invalid input shape: " + input_shape.ToString());
      }
      output_shape.push_back(output_dim_size);
    }
    return Status::OK();
  }

  // Use this method when pads are to be made symmetrical (if they are asymmetric)
  // and to collect metadata regarding the portion of the output (with "adjusted" pads)
  // to be sliced off to make the output correspond to the "actual" asymmetric paddings
  Status InferOutputShapeWithAdjustedPads(const TensorShape& input_shape,
                                          const std::vector<int64_t>& kernel_shape,
                                          const std::vector<int64_t>& strides_p,
                                          const std::vector<int64_t>& dilations_p,
                                          std::vector<int64_t>& pads_p,
                                          std::vector<int64_t>& output_shape,
                                          std::vector<int64_t>& output_shape_with_revised_pads,
                                          bool& post_slicing_needed,
                                          std::vector<int64_t>& slice_starts,
                                          std::vector<int64_t>& slice_ends,
                                          std::vector<int64_t>& slice_axes) const {
    size_t rank = input_shape.NumDimensions();
    // Make sure all "metadata" containers have the right number of elements
    if (rank > strides_p.size())
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Not enough elements in strides. Expected: ", rank, " Got: ", strides_p.size());

    if (rank > kernel_shape.size())
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Not enough elements in kernel shape. Expected: ", rank, " Got: ", kernel_shape.size());

    if (rank > dilations_p.size())
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Not enough elements in dilations. Expected: ", rank, " Got: ", dilations_p.size());

    if ((2 * rank) > pads_p.size())
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Not enough elements in pads. Expected: ", (2 * rank), " Got: ", pads_p.size());

    for (size_t dim = 0; dim < rank; ++dim) {
      int64_t output_dim_size = 0;
      ORT_RETURN_IF_ERROR(ComputePadAndOutputShape(input_shape[dim],
                                                   strides_p[dim],
                                                   kernel_shape[dim],
                                                   dilations_p[dim],
                                                   auto_pad,
                                                   pads_p[dim],
                                                   pads_p[rank + dim],
                                                   output_dim_size));
      if (output_dim_size <= 0) {
        return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Invalid input shape: " + input_shape.ToString());
      }

      // This is the "actual" output shape of the Conv op (i.e.) with given pads as is
      output_shape.push_back(output_dim_size);

      // Deal with asymmetric pads if any and adjust them to be symmetric
      // Along the way - note down how many values need to be sliced out from the start and end
      // of each spatial dimension while we over-pad to get symmetric pads

      if (pads_p[dim] == pads_p[rank + dim]) {
        // symmetric padding - No operation as such
        // Make note of the dim size of the output (to be used if there are other symmetrically padded dims)
        output_shape_with_revised_pads.push_back(output_dim_size);
      } else {
        // asymmetric padding

        int64_t& pad_head = pads_p[dim];
        int64_t& pad_tail = pads_p[rank + dim];
        int64_t stride = strides_p[dim];

        bool head_overpadded = false;

        if (pad_head < pad_tail) {
          int64_t excess_output_head = 0;

          // pad_head is under-padded, so "adjust" it by adding more padding
          while (pad_head < pad_tail) {
            // keep over-padding in multiples of 'strides' so that
            // the filter slides over correctly

            pad_head += stride;
            excess_output_head += 1;  // each multiple of stride contributes to 1 excess output value
          }

          post_slicing_needed = true;
          slice_axes.push_back(dim + 2);
          slice_starts.push_back(excess_output_head);
          slice_ends.push_back(excess_output_head + output_dim_size);                      // we may modify this below
          output_shape_with_revised_pads.push_back(excess_output_head + output_dim_size);  // we may modify this below
          head_overpadded = true;
        }

        // we may enter this section even if the head was initially under-padded,
        // because we had to over-pad by multiples of 'stride', now `pad_head` might be > `pad_tail`
        if (pad_tail < pad_head) {
          pad_tail = pad_head;
          auto revised_dim_size = ComputeOutputShape(input_shape[dim], strides_p[dim],
                                                     kernel_shape[dim], dilations_p[dim],
                                                     pad_head, pad_tail);

          if (head_overpadded) {
            // Head has already been over-padded
            // Additional tail pads need not result in additional output
            // Ensure that the size has changed - otherwise no operation needed.
            if (revised_dim_size !=
                output_shape_with_revised_pads[output_shape_with_revised_pads.size() - 1]) {
              output_shape_with_revised_pads[output_shape_with_revised_pads.size() - 1] = revised_dim_size;
            }
          } else {
            // Additional tail pads need not result in additional output
            // Ensure that the size has changed - otherwise no operation needed.
            if (revised_dim_size != output_dim_size) {
              // Head has not been over-padded. Only tail pads need to be modified.
              post_slicing_needed = true;

              slice_axes.push_back(dim + 2);
              slice_starts.push_back(0);
              slice_ends.push_back(output_dim_size - revised_dim_size);
            }

            // make note of the shape of this spatial dimension
            output_shape_with_revised_pads.push_back(revised_dim_size);
          }
        }
      }
    }
    return Status::OK();
  }

  bool HasStridesOneAndNoPadding() const {
    if (std::all_of(strides.begin(), strides.end(), [](int64_t v) { return v == 1; })) {
      if (std::all_of(pads.begin(), pads.end(), [](int64_t v) { return v == 0; })) {
        return true;
      }
    }
    return false;
  }

  AutoPadType auto_pad = AutoPadType::NOTSET;
  int64_t group;
  bool kernel_shape_specified;
  std::vector<int64_t> strides;
  std::vector<int64_t> pads;
  std::vector<int64_t> dilations;
  std::string activation;
  float alpha;

 private:
  std::vector<int64_t> kernel_shape_;  // must use ComputeKernelShape(...), instead of kernel_shape_
};

}  // namespace onnxruntime
