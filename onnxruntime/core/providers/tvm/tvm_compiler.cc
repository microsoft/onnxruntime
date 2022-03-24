// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <utility>

#include "tvm_compiler.h"
#include "tvm_api.h"


namespace onnxruntime {
namespace tvm {

TVMCompiler::TVMCompiler(std::string&& onnx_model_str,
                         const std::string& model_path,
                         int opset) :
onnx_model_str_(std::move(onnx_model_str)),
model_path_(model_path),
opset_(opset) {
}

auto TVMCompiler::operator()(const TvmEPOptions& options,
                             const TVMTensorShapes& input_shapes) -> ModulePtr {
  if (mod_) {
    return mod_;
  }

  mod_ = std::make_shared<TvmModule>();
  *mod_ = tvm::TVMCompile(onnx_model_str_,
                          model_path_,
                          options,
                          opset_,
                          input_shapes);
  onnx_model_str_.clear();
  return mod_;
}

}   // namespace tvm
}   // namespace onnxruntime
