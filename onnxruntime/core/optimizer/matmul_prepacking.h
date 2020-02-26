// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"
#include "core/providers/cpu/math/matmul_prepacked.h"

struct MLAS_GEMM_PARAMETERS;

namespace onnxruntime {

NodeAttributes GemmParamsToNodeAttributes(const MLAS_GEMM_PARAMETERS& params);

template<typename Impl>
Status GemmParamsFromNodeAttributes(const OpNodeProtoHelper<Impl>& node_context, MLAS_GEMM_PARAMETERS& params);

/**
@Class MatMulPrepacking
Replace MatMul(A, B) with MatMulPrepacked(A, PackForGemm(B)) when B is a constant.
This saves runtime cost of 'packing' in MLAS.
*/
class MatMulPrepacking : public RewriteRule {
public:
  explicit MatMulPrepacking(int max_num_threads)
    : RewriteRule("MatMulPrepacking"),
      max_num_threads_(max_num_threads) {
  }

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"MatMul"};
  }

private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;
  common::Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;

  int max_num_threads_;
};

} // namespace onnxruntime
