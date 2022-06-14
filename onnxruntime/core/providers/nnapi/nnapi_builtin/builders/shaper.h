// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/common/status.h"

namespace onnxruntime {
namespace nnapi {

class Shaper {
 public:
  using Shape = std::vector<uint32_t>;

  void AddShape(const std::string& name, const Shape& shape);
  inline const Shape& operator[](const std::string& key) const {
    return shape_map_.at(key);
  }

  common::Status Conv(const std::string& input_name,
                      const std::string& weight_name,
                      const std::vector<int32_t>& onnx_pads,
                      const std::vector<int32_t>& onnx_strides,
                      const std::vector<int32_t>& onnx_dilations,
                      bool nchw,
                      const std::string& output_name);

  common::Status DepthwiseConv(const std::string& input_name,
                               const std::string& weight_name,
                               const std::vector<int32_t>& onnx_pads,
                               const std::vector<int32_t>& onnx_strides,
                               const std::vector<int32_t>& onnx_dilations,
                               bool nchw,
                               const std::string& output_name);

  common::Status Pool(const std::string& input_name,
                      const std::vector<int32_t>& onnx_pads,
                      const std::vector<int32_t>& onnx_strides,
                      const std::vector<int32_t>& kernel_shape,
                      bool nchw,
                      const std::string& output_name);

  common::Status Reshape(const std::string& input_name, const std::vector<int32_t>& shape, const std::string& output_name);

  common::Status Transpose(const std::string& input_name, const std::vector<int32_t>& perm, const std::string& output_name);

  common::Status Eltwise(const std::string& input1_name, const std::string& input2_name, const std::string& output_name);

  common::Status Identity(const std::string& input_name, const std::string& output_name);

  common::Status FC(const std::string& input1_name, const std::string& input2_name, const std::string& output_name);

  common::Status Concat(const std::vector<std::string>& input_names, const int32_t axis, const std::string& output_name);

  common::Status Squeeze(const std::string& input_name, const std::vector<int32_t>& axes, const std::string& output_name);

  common::Status DepthToSpace(const std::string& input_name,
                              const int32_t blocksize,
                              bool nchw,
                              const std::string& output_name);

  common::Status ResizeUsingScales(const std::string& input_name,
                                   const float scale_h, const float scale_w,
                                   bool nchw,
                                   const std::string& output_name);
  common::Status ResizeUsingOutputSizes(const std::string& input_name,
                                        const uint32_t output_h, const uint32_t output_w,
                                        bool nchw,
                                        const std::string& output_name);

  // If the shape of certain input is dynamic
  // Use the following 2 functions to update the particular shape
  // and calculate the new output shape
  // Only perform this when the NNAPI model is finalized!
  common::Status UpdateShape(const std::string& name, const Shape& new_shape);
  common::Status UpdateDynamicDimensions();

  void Clear();

 private:
  common::Status ConvImpl(const std::string& input_name,
                          const std::string& weight_name,
                          const std::vector<int32_t>& onnx_pads,
                          const std::vector<int32_t>& onnx_strides,
                          const std::vector<int32_t>& onnx_dilations,
                          bool nchw,
                          const std::string& output_name);

  common::Status DepthwiseConvImpl(const std::string& input_name,
                                   const std::string& weight_name,
                                   const std::vector<int32_t>& onnx_pads,
                                   const std::vector<int32_t>& onnx_strides,
                                   const std::vector<int32_t>& onnx_dilations,
                                   bool nchw,
                                   const std::string& output_name);

  common::Status PoolImpl(const std::string& input_name,
                          const std::vector<int32_t>& onnx_pads,
                          const std::vector<int32_t>& onnx_strides,
                          const std::vector<int32_t>& kernel_shape,
                          bool nchw,
                          const std::string& output_name);

  common::Status ReshapeImpl(const std::string& input_name, const std::vector<int32_t>& shape, const std::string& output_name);
  common::Status TransposeImpl(const std::string& input_name, const std::vector<int32_t>& perm, const std::string& output_name);
  common::Status EltwiseImpl(const std::string& input1_name, const std::string& input2_name, const std::string& output_name);
  common::Status IdentityImpl(const std::string& input_name, const std::string& output_name);
  common::Status FCImpl(const std::string& input1_name, const std::string& input2_name, const std::string& output_name);
  common::Status ConcatImpl(const std::vector<std::string>& input_names, const int32_t axis, const std::string& output_name);
  common::Status SqueezeImpl(const std::string& input_names, const std::vector<int32_t>& axes, const std::string& output_name);
  common::Status DepthToSpaceImpl(const std::string& input_names,
                                  const int32_t blocksize,
                                  bool nchw,
                                  const std::string& output_name);
  common::Status ResizeUsingScalesImpl(const std::string& input_name,
                                       const float scale_h, const float scale_w,
                                       bool nchw,
                                       const std::string& output_name);
  common::Status ResizeUsingOutputSizesImpl(const std::string& input_name,
                                            const uint32_t output_h, const uint32_t output_w,
                                            bool nchw,
                                            const std::string& output_name);

  std::unordered_map<std::string, Shape> shape_map_;
  std::vector<std::function<common::Status(Shaper&)>> shape_ops_;
};

}  // namespace nnapi
}  // namespace onnxruntime
