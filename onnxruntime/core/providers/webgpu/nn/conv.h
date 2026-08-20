// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

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

template <bool is_channels_last, bool is_fused>
class Conv : public WebGpuKernel {
 public:
#if !defined(__wasm__)
  // Abstract base class for alternative optimized Conv implementations (currently the
  // subgroup-matrix path). Implementations hold the parent Conv and read
  // conv_attrs/activation/... from it, mirroring MatMul::MatMulOptImpl.
  class ConvOptImpl {
   public:
    explicit ConvOptImpl(const Conv& parent) : parent_(parent) {}
    virtual ~ConvOptImpl() = default;

    // Attempts the optimized path, reading the Conv operands from `context` and the
    // Conv attributes from the parent. Sets handled=true when it ran; leaves
    // handled=false (allocating nothing) so the caller falls back to the normal Conv
    // path. Called before ComputeInternal has resolved auto_pad or allocated the
    // output, so an implementation that needs those runs its own shape inference.
    virtual Status Compute(ComputeContext& context, /*out*/ bool& handled) = 0;

   protected:
    const Conv& parent_;
  };
#endif

  Conv(const OpKernelInfo& info) : WebGpuKernel(info), conv_attrs_(info) {
    if (is_fused) {
      ORT_ENFORCE(GetFusedActivationAttr(info, activation_).IsOK());
    }
    // Whether the weight input (index 1) is a constant initializer. An optimized
    // implementation uses this to decide whether it may cache a derived form of the
    // weight (e.g. a transposed copy) across Runs; a non-constant weight can change
    // between Runs, so anything derived from it must be rebuilt every time.
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

 private:
  // Alternative optimized implementation (lazily created on the first Compute call,
  // once the device capabilities can be queried from the compute context). A null
  // impl_ after initialization means this device has no optimized path.
#if !defined(__wasm__)
  mutable std::unique_ptr<ConvOptImpl> impl_;
  mutable std::once_flag impl_init_flag_;
#endif
};

Status TransposeKernel(ComputeContext& context, const Tensor* kernel, const TensorShape& kernel_shape, Tensor* transposed_kernel, const InlinedVector<size_t>& perm);

}  // namespace webgpu
}  // namespace onnxruntime
