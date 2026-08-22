// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>

#include "core/common/optional.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/nn/fuse_utils.h"

namespace onnxruntime {
namespace webgpu {

// Eligibility for the MatMul Conv fast paths, which reshape a Conv into a plain 2D
// matmul (see Conv::ComputeInternal and the subgroup-matrix Conv impl). A reshape is
// only equivalent to the Conv when nothing is padded: a leading pad shifts the window,
// and a trailing pad adds output positions the matmul never computes. Comparing the
// inferred output geometry on top of that also covers dilations, which change the
// effective kernel size without appearing anywhere in the reshape.
// The containers hold the spatial attributes in ONNX order, i.e. pads is
// [begin..., end...] and strides is one entry per spatial dim; both must already be
// defaulted/normalized to the 2D layout the callers use.

// True when every pad is zero, leading and trailing, across all spatial dims.
template <typename PadContainer>
bool HasNoConvPadding(const PadContainer& pads) {
  return std::all_of(pads.begin(), pads.end(), [](auto p) { return p == 0; });
}

// 1x1 reshape: each N,H,W position becomes an independent matmul row, so the output
// grid has to match the input grid one-to-one in row-major order. With a 1x1 kernel, no
// padding and unit strides that holds for any dilation (the effective kernel stays
// 1x1), so this predicate needs only the attributes -- PrePackInternal, which cannot
// see the input shape, uses the same one to stay in sync with ComputeInternal.
template <typename PadContainer, typename StrideContainer>
bool IsConv1x1MatMul(int64_t kernel_height, int64_t kernel_width,
                     const PadContainer& pads, const StrideContainer& strides) {
  return kernel_height == 1 && kernel_width == 1 && HasNoConvPadding(pads) &&
         strides[0] == 1 && strides[1] == 1;
}

// same-size reshape: the whole window folds into a single matmul row per batch element,
// so the kernel must cover the entire input and leave exactly one output position.
// Only valid for channels-last (the caller checks that).
template <typename PadContainer>
bool IsConvSameSizeMatMul(int64_t input_height, int64_t input_width,
                          int64_t kernel_height, int64_t kernel_width,
                          int64_t output_height, int64_t output_width,
                          const PadContainer& pads) {
  return input_height == kernel_height && input_width == kernel_width &&
         output_height == 1 && output_width == 1 && HasNoConvPadding(pads);
}

template <bool is_channels_last, bool is_fused>
class Conv : public WebGpuKernel {
 public:
#if !defined(__wasm__)
  // Abstract base class for alternative optimized Conv implementations (currently the
  // subgroup-matrix 1x1 / same-size path). Implementations hold the parent Conv and read
  // conv_attrs/activation/... from it, mirroring MatMul::MatMulOptImpl.
  class ConvOptImpl {
   public:
    explicit ConvOptImpl(const Conv& parent) : parent_(parent) {}
    virtual ~ConvOptImpl() = default;

    // Attempts the optimized path, reading the Conv operands from `context` and the Conv
    // attributes from the parent. Sets handled=true when it ran; leaves handled=false
    // (allocating nothing) so the caller falls back to the normal Conv path.
    virtual Status Compute(ComputeContext& context, /*out*/ bool& handled) = 0;

   protected:
    const Conv& parent_;
  };
#endif

  Conv(const OpKernelInfo& info) : WebGpuKernel(info), conv_attrs_(info) {
    if (is_fused) {
      ORT_ENFORCE(GetFusedActivationAttr(info, activation_).IsOK());
    }
    // Whether the weight input (index 1) is a constant initializer. The 1x1 Conv
    // subgroup-matrix path uses this to decide it can pad an odd-N weight once and
    // cache the result; a non-constant weight must not be cached.
    const Tensor* w = nullptr;
    w_is_constant_ = info.TryGetConstantInput(1, &w);
  }
  Status ComputeInternal(ComputeContext& context) const override;

  Status PrePackInternal(ComputeContextBase& context,
                         const Tensor& tensor,
                         int input_idx,
                         AllocatorPtr alloc,
                         /*out*/ bool& is_packed) override;

  // State an optimized implementation (ConvOptImpl) reads from its parent Conv.
  const ConvAttributes& ConvAttrs() const { return conv_attrs_; }
  const Activation& ConvActivation() const { return activation_; }
  // True when input 1 (the weight) is a constant initializer. See w_is_constant_.
  bool IsWeightConstant() const { return w_is_constant_; }
  // The prepacked (OIHW->HWIO transposed) weight, or null when it is not prepacked.
  const Tensor* PrepackedKernel() const { return transposed_kernel_.get(); }

 protected:
  ConvAttributes conv_attrs_;
  Activation activation_;
  std::unique_ptr<Tensor> transposed_kernel_;  // should only have value when `is_initializer` AND `is_4D` AND `is_NHWC`
  bool w_is_constant_ = false;                 // whether the weight input (index 1) is a constant initializer

  // Optional subgroup-matrix implementation for the 1x1 Conv path, lazily created
  // on the first Compute call (once device capabilities can be queried). Null after
  // initialization means the device has no subgroup-matrix support.
#if !defined(__wasm__)
  mutable std::unique_ptr<ConvOptImpl> impl_;
  mutable std::once_flag impl_init_flag_;
#endif
};

Status TransposeKernel(ComputeContext& context, const Tensor* kernel, const TensorShape& kernel_shape, Tensor* transposed_kernel, const InlinedVector<size_t>& perm);

}  // namespace webgpu
}  // namespace onnxruntime
