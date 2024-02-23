// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <utility>

#include "tvm_compiler.h"
#include "tvm_api.h"

namespace onnxruntime {
namespace tvm {

auto TVMCompilerBase::operator()(const TvmEPOptions& options,
                                 const TVMTensorShapes& input_shapes) -> ModulePtr {
  if (mod_) {
    return mod_;
  }

  mod_ = std::make_shared<TvmModule>();
  this->compileTVMModule(options, input_shapes);

  return mod_;
}

TVMCompiler::TVMCompiler(std::string&& onnx_model_str,
                         const std::string& model_path,
                         int opset) : onnx_model_str_(std::move(onnx_model_str)),
                                      model_path_(model_path),
                                      opset_(opset) {
}

void TVMCompiler::compileTVMModule(const TvmEPOptions& options,
                                   const TVMTensorShapes& input_shapes) {
  *mod_ = tvm::TVMCompile(options,
                          onnx_model_str_,
                          model_path_,
                          opset_,
                          input_shapes);

  onnx_model_str_.clear();
}

void TVMSoCompiler::compileTVMModule(const TvmEPOptions& options,
                                     [[maybe_unused]] const TVMTensorShapes& input_shapes) {
  *mod_ = tvm::TVMSoCompile(options);
}

}  // namespace tvm
}  // namespace onnxruntime
