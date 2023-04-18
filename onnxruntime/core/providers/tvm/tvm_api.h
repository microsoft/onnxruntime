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

TvmModule TVMCompile(const TvmEPOptions& options,
                     const std::string& onnx_txt,
                     const std::string& model_path,
                     int opset,
                     const TVMTensorShapes& input_shapes);
TvmModule TVMSoCompile(const TvmEPOptions& options);

void TVMSetInputs(TvmModule& mod, std::vector<size_t>& inds, std::vector<DLTensor>& inputs);
void TVM_VM_SetInputs(TvmModule& mod, std::vector<size_t>& inds, std::vector<DLTensor>& inputs);
void TVMSetOutputsZeroCopy(TvmModule& mod, std::vector<DLTensor>& outputs);
void TVM_VM_SetOutputsZeroCopy(TvmModule& mod, std::vector<DLTensor>& outputs);
void TVMGetOutputs(TvmModule& mod, std::vector<DLTensor>& outputs);
void TVM_VM_GetOutputs(TvmModule& mod, std::vector<DLTensor>& outputs);
void TVMGetOutputShapes(TvmModule& mod,
                        TVMTensorShapes& output_shapes);
void TVMRun(TvmModule& mod);
void TVM_VM_Run(TvmModule& mod);

}  // namespace tvm
}  // namespace onnxruntime

#endif  // TVM_API_H
