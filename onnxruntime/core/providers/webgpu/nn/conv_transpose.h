// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/common.h"

#include "core/providers/cpu/nn/conv_transpose_attributes.h"
#include "core/providers/webgpu/webgpu_kernel.h"
namespace onnxruntime {
namespace webgpu {

template <bool is_channels_last>
class ConvTranspose final : public WebGpuKernel {
 public:
  ConvTranspose(const OpKernelInfo& info) : WebGpuKernel(info), conv_transpose_attrs_(info) {
  }
  Status ComputeInternal(ComputeContext& context) const override;
  Status PrePackInternal(ComputeContextBase& context,
                         const Tensor& tensor,
                         int input_idx,
                         AllocatorPtr alloc,
                         /*out*/ bool& is_packed) override;

 protected:
  ConvTransposeAttributes conv_transpose_attrs_;

 private:
  enum class WeightLayout {
    ONNX,   // Unpacked weights: [C_in, C_out/group, spatial kernel dimensions...].
    HWIO,   // Prepacked 1D/2D weights: [kH, kW, C_in, C_out/group]; kH is 1 for 1D.
    DHWOI,  // Prepacked 3D weights: [kD, kH, kW, C_out/group, C_in].
  };

  std::unique_ptr<Tensor> prepacked_filter_;
  TensorShape original_filter_shape_;
  WeightLayout weight_layout_{WeightLayout::ONNX};
};

}  // namespace webgpu
}  // namespace onnxruntime
