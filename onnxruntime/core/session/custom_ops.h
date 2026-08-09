// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <gsl/span>
#include <memory>

#include "core/common/status.h"

struct OrtCustomOpDomain;
namespace onnxruntime {
class CustomRegistry;

common::Status CreateCustomRegistry(gsl::span<OrtCustomOpDomain* const> op_domains,
                                    std::shared_ptr<CustomRegistry>& output);

#if !defined(ORT_MINIMAL_BUILD)
class Graph;
class KernelTypeStrResolver;

namespace standalone {
// Register the schemas from any custom ops using the standalone invoker to call ORT kernels via OrtApi CreateOp.
// This is required so they can be captured when saving to an ORT format model.
// Implemented in standalone_op_invoker.cc
common::Status RegisterCustomOpNodeSchemas(KernelTypeStrResolver& kernel_type_str_resolver, Graph& graph);
}  // namespace standalone
#endif

}  // namespace onnxruntime
