// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef TVM_API_H
#define TVM_API_H

#include "tvm_common.h"

namespace onnxruntime {
namespace tvm {
    TvmModule TVMCompile(const std::string& onnx_txt,
                                    const std::string& model_path,
                                    const std::string& target,
                                    const std::string& target_host,
                                    int opt_level,
                                    int opset,
                                    bool freeze_params,
                                    const std::vector<std::vector<int64_t>>& input_shapes,
                                    bool nhwc = false,
                                    const std::string& tuning_logfile = "",
                                    const std::string& tuning_type = "AutoTVM");
    void TVMSetInputs(TvmModule& mod, std::vector<size_t>& inds, std::vector<DLTensor>& inputs);
    void TVMGetOutputShapes(TvmModule& mod,
                            size_t num_outputs,
                            std::vector<std::vector<int64_t>>& output_shapes);
    void TVMRun(TvmModule& mod, std::vector<DLTensor>& outputs, ::tvm::runtime::TVMRetValue *ret);
}  // namespace tvm
}  // namespace onnxruntime

#endif  // TVM_API_H