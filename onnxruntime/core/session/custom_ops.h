// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <gsl/span>
#include <memory>

#include "core/common/status.h"

struct OrtCustomOpDomain;
namespace onnxruntime {

struct CustomOpKernel : OpKernel {
  CustomOpKernel(const OpKernelInfo& info, const OrtCustomOp& op);

  ~CustomOpKernel() override {
    op_.KernelDestroy(op_kernel_);
  }

  Status Compute(OpKernelContext* ctx) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CustomOpKernel);

  const OrtCustomOp& op_;
  void* op_kernel_;
};

class CustomRegistry;

common::Status CreateCustomRegistry(gsl::span<OrtCustomOpDomain* const> op_domains,
                                    std::shared_ptr<CustomRegistry>& output);

#if !defined(ORT_MINIMAL_BUILD)
class Graph;
class KernelTypeStrResolver;
KernelCreateInfo CreateKernelCreateInfo(const std::string& domain, const OrtCustomOp* op, int version_from = 1, int version_to = INT_MAX);

namespace standalone {
// Register the schemas from any custom ops using the standalone invoker to call ORT kernels via OrtApi CreateOp.
// This is required so they can be captured when saving to an ORT format model.
// Implemented in standalone_op_invoker.cc
common::Status RegisterCustomOpNodeSchemas(KernelTypeStrResolver& kernel_type_str_resolver, Graph& graph);
}  // namespace standalone
#endif

}  // namespace onnxruntime
