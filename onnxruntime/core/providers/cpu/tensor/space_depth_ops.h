// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {

class SpaceDepthBase {
 protected:
  SpaceDepthBase(const OpKernelInfo& info) {
    ORT_ENFORCE(info.GetAttr("blocksize", &blocksize_).IsOK(),
                "Attribute blocksize is not set.");
  }

  Status InputValidationsAndOutputDims(const Tensor& input,
                                       int64_t& batch,
                                       int64_t& input_depth, int64_t& input_height, int64_t& input_width,
                                       int64_t& output_depth, int64_t& output_height, int64_t& output_width,
                                       bool is_space_to_depth) const {
    const TensorShape& input_shape = input.Shape();

    if (input_shape.NumDimensions() != 4) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "SpaceDepth ops require a 4-D input. Provided rank: ",
                             input_shape.NumDimensions());
    }

    batch = input_shape[0];
    input_depth = input_shape[1];
    input_height = input_shape[2];
    input_width = input_shape[3];

    if (is_space_to_depth) {  // SpaceToDepth op
      if ((input_height % this->blocksize_) != 0) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "SpaceToDepth requires input height to be a multiple of block_size");
      }

      if ((input_width % this->blocksize_) != 0) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "SpaceToDepth requires input width to be a multiple of block_size");
      }

      output_depth = input_depth * blocksize_ * blocksize_;
      output_height = input_height / blocksize_;
      output_width = input_width / blocksize_;

    } else {  // DepthToSpace op
      if ((input_depth % (blocksize_ * blocksize_) != 0)) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "DepthToSpace requires input depth to be a multiple of (block_size * blok_size)");
      }

      output_depth = input_depth / blocksize_ / blocksize_;
      output_height = input_height * blocksize_;
      output_width = input_width * blocksize_;
    }

    return Status::OK();
  }

  int64_t blocksize_;
};

class SpaceToDepth final : public OpKernel, SpaceDepthBase {
 public:
  SpaceToDepth(const OpKernelInfo& info) : OpKernel(info), SpaceDepthBase(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

class DepthToSpace final : public OpKernel, SpaceDepthBase {
 public:
  DepthToSpace(const OpKernelInfo& info) : OpKernel(info), SpaceDepthBase(info) {
    std::string mode;
    // if  mode doesn't exist, then it is the default "DCR" mode
    // (or) it is an opset < 11 model for which the only mode is "DCR" mode
    if (info.GetAttr("mode", &mode).IsOK()) {
      if (mode == "CRD")
        is_dcr_ = false;

      else if (mode != "DCR")
        ORT_THROW("DepthToSpace op: only 'DCR' and 'CRD' modes are supported");
    }
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  bool is_dcr_ = true;
};

}  //namespace onnxruntime
