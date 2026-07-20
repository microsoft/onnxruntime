// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#if !defined(SHARED_PROVIDER) && !defined(BUILD_CUDA_EP_AS_PLUGIN)
#include "core/framework/op_kernel.h"
#endif

namespace onnxruntime {

namespace space_depth_internal {

template <typename KernelInfoType>
inline int64_t ReadBlocksize(const KernelInfoType& info) {
  int64_t blocksize = 0;
  ORT_ENFORCE(info.template GetAttr<int64_t>("blocksize", &blocksize).IsOK(),
              "Attribute blocksize is not set.");
  return blocksize;
}

template <typename KernelInfoType>
inline bool ReadIsDCR(const KernelInfoType& info) {
  bool is_dcr = true;
  std::string mode;
  // If mode doesn't exist, then it is the default "DCR" mode
  // (or) it is an opset < 11 model for which the only mode is "DCR" mode.
  if (info.GetAttr("mode", &mode).IsOK()) {
    if (mode == "CRD") {
      is_dcr = false;
    } else if (mode != "DCR") {
      ORT_THROW("DepthToSpace op: only 'DCR' and 'CRD' modes are supported");
    }
  }

  return is_dcr;
}

template <bool IsNHWC = false>
inline Status InputValidationsAndOutputDimsCalc(int64_t blocksize,
                                                const Tensor& input,
                                                int64_t& batch,
                                                int64_t& input_depth, int64_t& input_height, int64_t& input_width,
                                                int64_t& output_depth, int64_t& output_height, int64_t& output_width,
                                                bool is_space_to_depth) {
  const TensorShape& input_shape = input.Shape();

  if (input_shape.NumDimensions() != 4) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "SpaceDepth ops require a 4-D input. Provided rank: ",
                           input_shape.NumDimensions());
  }

  batch = input_shape[0];
  if constexpr (IsNHWC) {
    input_depth = input_shape[3];
    input_height = input_shape[1];
    input_width = input_shape[2];
  } else {
    input_depth = input_shape[1];
    input_height = input_shape[2];
    input_width = input_shape[3];
  }

  if (is_space_to_depth) {  // SpaceToDepth op
    if ((input_height % blocksize) != 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "SpaceToDepth requires input height to be a multiple of block_size");
    }

    if ((input_width % blocksize) != 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "SpaceToDepth requires input width to be a multiple of block_size");
    }

    output_depth = input_depth * blocksize * blocksize;
    output_height = input_height / blocksize;
    output_width = input_width / blocksize;

  } else {  // DepthToSpace op
    if ((input_depth % (blocksize * blocksize) != 0)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "DepthToSpace requires input depth to be a multiple of (block_size * block_size)");
    }

    output_depth = input_depth / blocksize / blocksize;
    output_height = input_height * blocksize;
    output_width = input_width * blocksize;
  }

  return Status::OK();
}

}  // namespace space_depth_internal

class SpaceDepthBase {
 protected:
  template <typename KernelInfoType>
  explicit SpaceDepthBase(const KernelInfoType& info) : blocksize_(space_depth_internal::ReadBlocksize(info)) {}

  template <bool IsNHWC = false>
  Status InputValidationsAndOutputDimsCalc(const Tensor& input,
                                           int64_t& batch,
                                           int64_t& input_depth, int64_t& input_height, int64_t& input_width,
                                           int64_t& output_depth, int64_t& output_height, int64_t& output_width,
                                           bool is_space_to_depth) const {
    return space_depth_internal::InputValidationsAndOutputDimsCalc<IsNHWC>(
        blocksize_, input, batch, input_depth, input_height, input_width,
        output_depth, output_height, output_width, is_space_to_depth);
  }

  int64_t blocksize_;
};

#if !defined(SHARED_PROVIDER) && !defined(BUILD_CUDA_EP_AS_PLUGIN)

class SpaceToDepth final : public OpKernel, SpaceDepthBase {
 public:
  explicit SpaceToDepth(const OpKernelInfo& info) : OpKernel(info), SpaceDepthBase(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

class DepthToSpace final : public OpKernel, SpaceDepthBase {
 public:
  explicit DepthToSpace(const OpKernelInfo& info)
      : OpKernel(info), SpaceDepthBase(info), is_dcr_(space_depth_internal::ReadIsDCR(info)) {}

  Status Compute(OpKernelContext* context) const override;

 private:
  bool is_dcr_ = true;
};

#endif  // !defined(SHARED_PROVIDER) && !defined(BUILD_CUDA_EP_AS_PLUGIN)

}  // namespace onnxruntime
