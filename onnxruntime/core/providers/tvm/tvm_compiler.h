// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef TVM_COMPILER_H
#define TVM_COMPILER_H

#include <string>
#include <memory>

#include "tvm_common.h"
#include "tvm_ep_options.h"


namespace onnxruntime {
namespace tvm {

class TVMCompiler {
  using ModulePtr = std::shared_ptr<TvmModule>;
public:
  TVMCompiler() = delete;
  ~TVMCompiler() = default;

  TVMCompiler(std::string&& onnx_model_str,
              const std::string& model_path,
              int opset);

  ModulePtr operator()(const TvmEPOptions& options,
                       const TVMTensorShapes& input_shapes);

private:
  ModulePtr mod_;
  std::string onnx_model_str_;
  std::string model_path_;
  int opset_;
};

}   // namespace tvm
}   // namespace onnxruntime

#endif  // TVM_COMPILER_H
