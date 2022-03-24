// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef TVM_API_H
#define TVM_API_H

#include <vector>
#include <string>

#include "tvm_common.h"
#include "tvm_defaults.h"
#include "tvm_ep_options.h"


namespace onnxruntime {
namespace tvm {

  TvmModule TVMCompile(const std::string& onnx_txt,
                       const std::string& model_path,
                       const TvmEPOptions& options,
                       int opset,
                       const TVMTensorShapes& input_shapes);
  void TVMSetInputs(TvmModule& mod, std::vector<size_t>& inds, std::vector<DLTensor>& inputs);
  void TVM_VM_SetInputs(TvmModule& mod, std::vector<size_t>& inds, std::vector<DLTensor>& inputs);
  void TVMGetOutputs(TvmModule& mod, std::vector<DLTensor>& outputs);
  void TVM_VM_GetOutputs(TvmModule& mod, std::vector<DLTensor>& outputs);
  void TVMGetOutputShapes(TvmModule& mod,
                          TVMTensorShapes& output_shapes);
  void TVMRun(TvmModule& mod);
  void TVM_VM_Run(TvmModule& mod);

}  // namespace tvm
}  // namespace onnxruntime

#endif  // TVM_API_H
