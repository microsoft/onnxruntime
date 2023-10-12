#pragma once

#include "core/optimizer/rewrite_rule.h"


namespace onnxruntime
{
/*
*   This fusion submerges a BatchNormalization operator to it's super 
*   precedding MatMul operator, if and only if MatmulBNFusion::SatisfyCondition()
*   is true.
*/
class MatmulBNFusion : public RewriteRule
{
public:
    MatmulBNFusion() : RewriteRule("MatMul_BatchNormalization_Fusion")
    {

    }

    std::vector<std::string> TargetOpTypes() const noexcept
    {
      return {"MatMul"};
    }

private:
    bool SatisfyCondition(
        const Graph& graph,
        const Node& node,
        const logging::Logger& logger) const override;

    Status Apply(
        Graph& graph,
        Node& matmulNode,
        RewriteRuleEffect& rule_effect,
        const logging::Logger& logger) const override;

    bool MatchPath(
        const Node& parentNode,
        const gsl::span<std::pair<std::string, std::initializer_list<int>>>& path,
        const Node& childNode) const;
};
}