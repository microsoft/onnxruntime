// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/optimizer/cuda_runtime_fusion.h"

#if !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_CONTRIB_OPS)
#include "core/graph/contrib_ops/source_replacement.h"

namespace onnxruntime::contrib::cuda::fusion {

struct SourceReplacement {
  using DiscoverFunction = Status (*)(const SourceReplacement&, const GraphViewer&,
                                      const IExecutionProvider::IKernelLookup&, const logging::Logger&,
                                      std::vector<std::unique_ptr<ComputeCapability>>&);
  OpKey target;
  std::string rule_id;
  std::string config;
  SourceBody body;
  DiscoverFunction discover;

  ExecutableSource Metadata() const {
    // v0's variant is the rule name, not an independent support identifier.
    return {target, kCudaExecutionProvider, rule_id, &body};
  }
};

gsl::span<const SourceReplacement> ScaledSiLUSources();
gsl::span<const SourceReplacement> CausalConvWithStateSources();

// Derived only from the explicitly collected op companions; shared by selection and coverage.
gsl::span<const SourceReplacement* const> CudaSourceReplacements();
Status ValidateCudaSourceReplacements(gsl::span<const SourceReplacement* const> sources,
                                      gsl::span<const OpKey> schemas,
                                      gsl::span<const SourcePolicyRecord> policies,
                                      gsl::span<const OpKey> legacy);

// v0's CUDA adapter supports one target node; the common representation remains a DAG.
bool SupportsCudaReplacement(const rewrite::Replacement& recipe, size_t output_count);

}  // namespace onnxruntime::contrib::cuda::fusion
#endif
