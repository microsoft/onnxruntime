// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/codegen/common/common.h"
#include "core/common/common.h"
#include "core/graph/graph_viewer.h"

namespace onnxruntime {
namespace codegen {

// Base class of Analysis
template <typename INPUT_TYPE>
class AnalysisBase {
 public:
  AnalysisBase() {}

  AnalysisBase(const std::string& name)
      : name_(name) {}

  ~AnalysisBase() = default;

  virtual void Evaluate(INPUT_TYPE) = 0;

  const std::string& Name() const {
    return name_;
  }

 protected:
  std::string name_{"Unknown"};

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(AnalysisBase);
};

using OrtAnalysis = AnalysisBase<const onnxruntime::GraphViewer&>;

// Add Promote for OrtAnalysis
DYNAMIC_PROMOTE(OrtAnalysis)

}  // namespace codegen
}  // namespace onnxruntime
