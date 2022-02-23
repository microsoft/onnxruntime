// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "tvm_compiler.h"
#include "tvm_api.h"

namespace onnxruntime {
namespace tvm {

TVMCompiler::TVMCompiler() {
    mod_ = std::make_shared<TvmModule>();
}

auto TVMCompiler::getModule(const std::string& onnx_model_str,
                            const std::string& model_path,
                            const TvmEPOptions& options,
                            int opset,
                            const std::vector<std::vector<int64_t>>& input_shapes) -> ModulePtr {
    if (mod_) {
        return mod_;
    }

    *mod_ = tvm::TVMCompile(onnx_model_str,
                            model_path,
                            options,
                            opset,
                            input_shapes);
    return mod_;
}

}   // namespace tvm
}   // namespace onnxruntime
