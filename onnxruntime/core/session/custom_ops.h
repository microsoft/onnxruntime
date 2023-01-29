// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace onnxruntime {
common::Status CreateCustomRegistry(gsl::span<OrtCustomOpDomain* const> op_domains,
                                    std::shared_ptr<CustomRegistry>& output);

namespace standalone {
// Register the schemas from any custom ops using the standalone invoker to call ORT kernels.
// This is required so they can be captured when saving to an ORT format model.
// implemented in standalone_op_invoker.cc
common::Status RegisterCustomOpNodeSchemas(KernelTypeStrResolver& kernel_type_str_resolver, Graph& graph);
}  // namespace standalone

}  // namespace onnxruntime
