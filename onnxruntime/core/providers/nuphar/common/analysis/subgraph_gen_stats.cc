// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "output_alias_analysis.h"
#include "subgraph_gen_stats.h"
#include "use_count_analysis.h"

namespace onnxruntime {
namespace codegen {

// NOTE: This code is a copy of old GraphStats for the later refacotring
//       No need to review them before refactor step 11
// TODO: Rewrite corresponding passes after refactor step 11

// SubGraphStats has two analysis passes
// The first pass, offset as 0,  is UseCountAnalysis
// The second pass, offset as 1,  is OutputAliasAnalysis
constexpr int UseCountAnalysisOffset = 0;
constexpr int OutputAliasAnalysisOffset = 1;

// Constructor
SubGraphStats::SubGraphStats(
    const std::shared_ptr<ShapeExprContext>& shape_infernece)
    : OrtGraphStats("SubGraphNupharX86") {
  auto use_count_pass = std::make_shared<UseCountAnalysis>(shape_infernece);
  passes_.push_back(use_count_pass);

  auto output_alias_pass = std::make_shared<OutputAliasAnalysis>();
  passes_.push_back(output_alias_pass);
}

int SubGraphStats::NodeUseCount(const onnxruntime::Node* node) const {
  ORT_ENFORCE(passes_.size() > UseCountAnalysisOffset);
  return Promote<UseCountAnalysis>(passes_[UseCountAnalysisOffset])->NodeUseCount(node);
}

void SubGraphStats::EvaluateSingleNode(const onnxruntime::Node& node) {
  if (IsFusedNode(node)) {
    const Graph* subgraph = GetSubgraph(node);
    Evaluate(GraphViewer(*subgraph));
    ORT_ENFORCE(passes_.size() > UseCountAnalysisOffset);

    // Update output count
    // since subgraph->GetOutputs() might not count some outputs that have another consumer.
    // E.g. A_output_def == B_input_def, and B_output_def is in subgraph->GetOutputs().
    //      In this case A_output_def won't be in subgraph->GetOutputs()
    //      But both A_output_def and B_output_def can be in node.OutputDefs().
    auto& graph_outputs = subgraph->GetOutputs();
    for (auto def : node.OutputDefs()) {
      if (std::find(graph_outputs.begin(), graph_outputs.end(), def) == graph_outputs.end()) {
        Promote<UseCountAnalysis>(passes_[UseCountAnalysisOffset])->IncrementCount(def);
      }
    }

  } else {
    ORT_ENFORCE(passes_.size() > OutputAliasAnalysisOffset);
    Promote<OutputAliasAnalysis>(passes_[OutputAliasAnalysisOffset])->EvaluateSingleNode(node);
  }
}

bool SubGraphStats::IsOutputNode(const onnxruntime::Node* node) const {
  ORT_ENFORCE(passes_.size() > OutputAliasAnalysisOffset);
  return Promote<OutputAliasAnalysis>(passes_[OutputAliasAnalysisOffset])->IsOutputNode(node);
}

bool SubGraphStats::IsOutputAlias(const onnxruntime::Node* node) const {
  ORT_ENFORCE(passes_.size() > OutputAliasAnalysisOffset);
  return Promote<OutputAliasAnalysis>(passes_[OutputAliasAnalysisOffset])->IsOutputAlias(node);
}

const onnxruntime::NodeArg* SubGraphStats::SourceDefOfOutputAlias(const onnxruntime::NodeArg* node) const {
  ORT_ENFORCE(passes_.size() > OutputAliasAnalysisOffset);
  return Promote<OutputAliasAnalysis>(passes_[OutputAliasAnalysisOffset])->SourceDefOfOutputAlias(node);
}

}  // namespace codegen
}  // namespace onnxruntime
