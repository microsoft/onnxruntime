// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/codegen/common/common.h"
#include "core/common/common.h"
#include "core/graph/graph_viewer.h"
#include "core/providers/nuphar/common/nuphar_subgraph.h"

namespace onnxruntime {
namespace nuphar {

// abstract class for Analysis
template <typename INPUT_TYPE>
class AnalysisBase {
 public:
  AnalysisBase() {}

  AnalysisBase(const std::string& name)
      : name_(name) {}

  virtual ~AnalysisBase() = default;

  virtual void Evaluate(INPUT_TYPE) = 0;

  const std::string& Name() const {
    return name_;
  }

 protected:
  const std::string name_{"Unknown"};

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(AnalysisBase);
};

using OrtAnalysis = AnalysisBase<const onnxruntime::GraphViewer&>;
using NupharAnalysis = AnalysisBase<const onnxruntime::nuphar::NupharSubgraphUnit&>;

// Add Promote for OrtAnalysis and NupharAnalysis
DYNAMIC_PROMOTE(OrtAnalysis)
DYNAMIC_PROMOTE(NupharAnalysis)

}  // namespace nuphar
}  // namespace onnxruntime
