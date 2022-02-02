// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef STVM_API_H
#define STVM_API_H

#include "stvm_common.h"

namespace stvm {
    tvm::runtime::Module TVMCompile(const std::string& onnx_txt,
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
    void TVMSetInputs(tvm::runtime::Module& mod, std::vector<size_t>& inds, std::vector<DLTensor>& inputs);
    void TVMGetOutputShapes(tvm::runtime::Module& mod,
                            size_t num_outputs,
                            std::vector<std::vector<int64_t>>& output_shapes);
    void TVMRun(tvm::runtime::Module& mod, std::vector<DLTensor>& outputs, tvm::runtime::TVMRetValue *ret);
}  // namespace stvm

#endif  // STVM_API_H