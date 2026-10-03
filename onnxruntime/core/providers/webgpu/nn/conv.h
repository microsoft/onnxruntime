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
#include "core/providers/webgpu/math/matmul.h"
#include "core/providers/webgpu/nn/fuse_utils.h"

namespace onnxruntime {
namespace webgpu {

// Layout of the kernel tensor that ComputeInternal consumes. `OIHW` is the layout the
// Conv operator is defined with; the others are produced by PrePackInternal and are each
// understood by exactly one consumer, so the layout has to be tracked explicitly rather
// than inferred from which prepacked tensor happens to be present.
enum class KernelLayout {
  OIHW,  // No prepacked tensor -- the kernel is read straight from input 1.
  HWIO,  // Consumed by grouped conv, the 1x1/same_size MatMul path and Conv2dMM.
  OHWI,  // Consumed by the Im2ColMatMul path only.
};

template <bool is_channels_last, bool is_fused>
class Conv : public WebGpuKernel {
 public:
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
  // The prepacked weight, or null when it is not prepacked. Its layout is HWIO or OHWI,
  // as reported by kernel_layout_.
  const Tensor* PrepackedKernel() const { return prepacked_kernel_.get(); }

 protected:
  ConvAttributes conv_attrs_;
  Activation activation_;
  // Set by PrePackInternal; null when the kernel could not be prepacked (e.g. the weight
  // is not a constant initializer), in which case ComputeInternal reads input 1 instead.
  std::unique_ptr<Tensor> prepacked_kernel_;
  // Layout of the tensor ComputeInternal ends up consuming -- `prepacked_kernel_` when it
  // is set, otherwise input 1. Stays `OIHW` while `prepacked_kernel_` is null.
  KernelLayout kernel_layout_{KernelLayout::OIHW};
  mutable MatMulOptImplCache matmul_compute_cache_;

  bool w_is_constant_ = false;  // whether the weight input (index 1) is a constant initializer

 private:
  // Owns the alternative optimized implementation and builds it on first use,
  // mirroring MatMulOptImplCache. The impl cannot be built in the ctor because it
  // needs the device capabilities, which are only reachable through a compute
  // context; and the same Conv kernel can see concurrent Compute calls, so creation
  // is guarded by call_once. A null return after initialization means this device
  // has no optimized path. Nested rather than a free class like MatMulOptImplCache
  // only because ConvOptImpl is itself nested in this template.
  class ConvOptImplCache {
   public:
    ConvOptImplCache() = default;
    ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ConvOptImplCache);

    // Defined in conv.cc: the factory it calls is declared in subgroup_matrix_conv.h,
    // which includes this header.
    ConvOptImpl* GetOrCreate(const Conv& parent, const ComputeContextBase& context);

   private:
    std::once_flag init_flag_;
    std::unique_ptr<ConvOptImpl> impl_;
  };

  mutable ConvOptImplCache opt_impl_cache_;
};

Status TransposeKernel(ComputeContext& context, const Tensor* kernel, const TensorShape& kernel_shape, Tensor* transposed_kernel, const InlinedVector<size_t>& perm);

}  // namespace webgpu
}  // namespace onnxruntime
