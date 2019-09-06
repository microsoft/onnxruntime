// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/codegen/common/common.h"
#include "core/common/common.h"
#include "core/graph/graph_viewer.h"
#include "core/providers/nuphar/common/analysis/analysis.h"

#include "core/providers/nuphar/common/nuphar_subgraph.h"
// Base class of GraphStatsBase
// GraphStatsBase holds analysis results from a graph
// GraphStatsBase can hold multiple analyses

namespace onnxruntime {
namespace nuphar {

template <typename INPUT_TYPE>
class GraphStatsBase {
 public:
  GraphStatsBase(const std::string& name)
      : name_(name) {}

  GraphStatsBase() {}

  virtual ~GraphStatsBase() = default;

  // Evaluate all passes
  virtual void Evaluate(INPUT_TYPE graph) {
    for (auto& pass : passes_) {
      pass->Evaluate(graph);
    }
  }

  // Set passes externally
  void SetAllPasses(const std::vector<std::shared_ptr<AnalysisBase<INPUT_TYPE>>>& passes) {
    passes_.clear();
    for (auto& pass : passes) {
      passes_.push_back(pass);
    }
  }

  // Set existed evaluated passes externally
  void SetAllExistedEvaluatedPasses(
      const std::vector<std::shared_ptr<AnalysisBase<INPUT_TYPE>>>& passes) {
    existed_eval_passes_.clear();
    for (auto& pass : passes) {
      existed_eval_passes_.push_back(pass);
    }
  }

  const std::string& Name() const {
    return name_;
  }

 protected:
  const std::string name_{"Unknown"};

  std::vector<std::shared_ptr<AnalysisBase<INPUT_TYPE>>> passes_;

 private:
  // existed eval passes not requiring evaluation
  std::vector<std::shared_ptr<AnalysisBase<INPUT_TYPE>>> existed_eval_passes_;

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GraphStatsBase);
};

using OrtGraphStats = GraphStatsBase<const onnxruntime::GraphViewer&>;
using NupharSubgraphUnitStats = GraphStatsBase<const onnxruntime::nuphar::NupharSubgraphUnit&>;

// Add Promote for OrtGraphStats and NupharSubgraphUnitStats
DYNAMIC_PROMOTE(OrtGraphStats)
DYNAMIC_PROMOTE(NupharSubgraphUnitStats)

}  // namespace nuphar
}  // namespace onnxruntime
