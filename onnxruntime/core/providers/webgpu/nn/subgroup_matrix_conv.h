// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(__wasm__)

#include <memory>

#include "core/framework/tensor.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/nn/fuse_utils.h"

namespace onnxruntime {
namespace webgpu {

// Read-only view of the Conv kernel state an optimized Conv implementation reads.
// Conv<...> implements it so the impl can pull state from the parent (mirroring
// MatMul::MatMulOptImpl accessing MatMul via parent_) instead of receiving it as
// Compute arguments. Conv is a template, so this non-template interface is what the
// (non-template) impl holds a reference to.
class ConvOptImplParent {
 public:
  virtual ~ConvOptImplParent() = default;
  virtual const ConvAttributes& ConvAttrs() const = 0;
  virtual const Activation& ConvActivation() const = 0;
  virtual bool IsChannelsLast() const = 0;
  virtual bool IsWeightConstant() const = 0;
  // The prepacked (OIHW->HWIO transposed) weight, or null when it is not prepacked.
  virtual const Tensor* PrepackedKernel() const = 0;
};

// Abstract base for an optional optimized 1x1 / same-size Conv implementation. Holds the
// parent Conv (via ConvOptImplParent) so Compute reads conv_attrs/activation/... from it,
// mirroring the MatMul::MatMulOptImpl pattern.
class ConvOptImpl {
 public:
  explicit ConvOptImpl(const ConvOptImplParent& parent) : parent_(parent) {}
  virtual ~ConvOptImpl() = default;

  // Attempts the optimized path, reading the Conv operands from `context` and the Conv
  // attributes from the parent. Sets handled=true when it ran; leaves handled=false
  // (allocating nothing) so the caller falls back to the normal Conv path.
  virtual Status Compute(ComputeContext& context, /*out*/ bool& handled) = 0;

 protected:
  const ConvOptImplParent& parent_;
};

// Creates a subgroup-matrix Conv 1x1 implementation on devices whose vendor policy supports
// the subgroup-matrix kernel; returns nullptr otherwise, so the caller falls back to the
// normal Conv path.
std::unique_ptr<ConvOptImpl> CreateSubgroupMatrixConvImpl(const ConvOptImplParent& parent,
                                                             const ComputeContextBase& context);

}  // namespace webgpu
}  // namespace onnxruntime

#endif  // !defined(__wasm__)
