// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#ifndef BUILD_CUDA_EP_AS_PLUGIN
#include "core/providers/cpu/tensor/space_depth_ops.h"
#endif

namespace onnxruntime {
namespace cuda {

#ifdef BUILD_CUDA_EP_AS_PLUGIN
// PLUGIN BUILD ADAPTATION: SpaceDepthBase (in cpu/tensor/space_depth_ops.h)
// cannot be included because it pulls in core/framework/op_kernel.h which
// conflicts with the adapter types. This inline namespace reimplements the
// validation and dimension-calculation logic. Keep in sync with SpaceDepthBase.
namespace detail {

template <bool IsNHWC = false>
Status InputValidationsAndOutputDimsCalc(int64_t blocksize,
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

  if (is_space_to_depth) {
    if ((input_height % blocksize) != 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "SpaceToDepth requires input height to be a multiple of block_size");
    }
    if ((input_width % blocksize) != 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "SpaceToDepth requires input width to be a multiple of block_size");
    }
    output_depth = input_depth * blocksize * blocksize;
    output_height = input_height / blocksize;
    output_width = input_width / blocksize;
  } else {
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

}  // namespace detail
#endif  // BUILD_CUDA_EP_AS_PLUGIN

template <bool Layout>
class SpaceToDepth final : public CudaKernel
#ifndef BUILD_CUDA_EP_AS_PLUGIN
    ,
                           SpaceDepthBase
#endif
{
 public:
  explicit SpaceToDepth(const OpKernelInfo& info)
      : CudaKernel(info)
#ifndef BUILD_CUDA_EP_AS_PLUGIN
        ,
        SpaceDepthBase(info)
#endif
  {
#ifdef BUILD_CUDA_EP_AS_PLUGIN
    // Plugin builds cannot inherit from SpaceDepthBase, so extract the
    // blocksize attribute directly from OpKernelInfo.
    ORT_ENFORCE(info.GetAttr("blocksize", &blocksize_).IsOK(),
                "Attribute blocksize is not set.");
#endif
  }

  Status ComputeInternal(OpKernelContext* context) const override;

#ifdef BUILD_CUDA_EP_AS_PLUGIN
 protected:
  template <bool IsNHWC = false>
  Status InputValidationsAndOutputDimsCalc(const Tensor& input,
                                           int64_t& batch,
                                           int64_t& input_depth, int64_t& input_height, int64_t& input_width,
                                           int64_t& output_depth, int64_t& output_height, int64_t& output_width,
                                           bool is_space_to_depth) const {
    return detail::InputValidationsAndOutputDimsCalc<IsNHWC>(
        blocksize_, input, batch, input_depth, input_height, input_width,
        output_depth, output_height, output_width, is_space_to_depth);
  }

  int64_t blocksize_;
#endif
};

template <bool Layout>
class DepthToSpace final : public CudaKernel
#ifndef BUILD_CUDA_EP_AS_PLUGIN
    ,
                           SpaceDepthBase
#endif
{
 public:
  explicit DepthToSpace(const OpKernelInfo& info)
      : CudaKernel(info)
#ifndef BUILD_CUDA_EP_AS_PLUGIN
        ,
        SpaceDepthBase(info)
#endif
  {
#ifdef BUILD_CUDA_EP_AS_PLUGIN
    // Plugin builds cannot inherit from SpaceDepthBase, so extract the
    // blocksize attribute directly from OpKernelInfo.
    ORT_ENFORCE(info.GetAttr("blocksize", &blocksize_).IsOK(),
                "Attribute blocksize is not set.");
#endif
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

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  bool is_dcr_ = true;

#ifdef BUILD_CUDA_EP_AS_PLUGIN
 protected:
  template <bool IsNHWC = false>
  Status InputValidationsAndOutputDimsCalc(const Tensor& input,
                                           int64_t& batch,
                                           int64_t& input_depth, int64_t& input_height, int64_t& input_width,
                                           int64_t& output_depth, int64_t& output_height, int64_t& output_width,
                                           bool is_space_to_depth) const {
    return detail::InputValidationsAndOutputDimsCalc<IsNHWC>(
        blocksize_, input, batch, input_depth, input_height, input_width,
        output_depth, output_height, output_width, is_space_to_depth);
  }

  int64_t blocksize_;
#endif
};

}  // namespace cuda
}  // namespace onnxruntime
