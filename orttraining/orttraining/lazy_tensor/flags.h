// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
namespace onnxruntime {
namespace lazytensor {
// This file contains environment variables that control
// the behavior of ORT as LazyTensor's backend.
// Most variables are for debug purpose.
// Example:
//   LORT_CHECK_TENSOR_CONTENT=1 LORT_DUMP_GRAPH=1
//   LORT_DUMP_INPUTS_OUTPUTS=1 LORT_CHECK_BASELINE=1
//   LORT_RELATIVE_TOLERANCE=1e-3 python main.py

// When returing true, we dump the inputs and outputs
// when ORT (and Pytorch when ORTLTCHECKBASELINE is set to 1)
// executes the subgraph.
bool DumpInputsOutputs();
// Returns true to dump the torch::jit::Graph ORT receives
// from LazyTensor.
bool DumpGraph();
// If returned value is true, run torch::jit::GraphExecutor
// and compare its outputs with ORT's outputs.
// Only types and shapes are compared. The user can control
// the checking mechanism. For example, set
// LORT_CHECK_TENSOR_CONTENT=1 to compare tensor elements.
//
// Related functions' dependency graph:
//  CheckBaseline -> CheckTensorContent -> AbsoluteTolerance
//                                   '---> RelativeTolerance
// bool CheckBaseline();
std::string RunType();
// If this function returns true, all aten ops seen by ORT
// will be printed. We also tag if these are supported or not.
bool DumpAtenOpHistory();
// If this function returns true, check tensor's elements
// when CheckBaseline() returns true.
bool CheckTensorContent();
// The "absolute_tol" in
// |value-expected| <= |expected| * relative_tol + absolute_tol
double AbsoluteTolerance();
// The "relative_tol" in
// |value-expected| <= |expected| * relative_tol + absolute_tol
double RelativeTolerance();
bool DumpOnnxFusion();

class DynamicSettings {
 public:
  static DynamicSettings& GetInstance() {
    static DynamicSettings instance;
    return instance;
  }
  DynamicSettings(DynamicSettings const&) = delete;
  void operator=(DynamicSettings const&) = delete;
  bool GetOnnxFusionFlag() const {
    return onnx_fusion_status_;
  }
  void SetOnnxFusionFlag(bool status) {
    onnx_fusion_status_ = status;
  }

 private:
  DynamicSettings() : onnx_fusion_status_(true){};
  bool onnx_fusion_status_;
};

}  // namespace lazytensor
}  // namespace onnxruntime
