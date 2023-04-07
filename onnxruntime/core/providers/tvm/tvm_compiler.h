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

class TVMCompilerBase {
 public:
  using ModulePtr = std::shared_ptr<TvmModule>;

  TVMCompilerBase() = default;
  virtual ~TVMCompilerBase() = default;

  ModulePtr operator()(const TvmEPOptions& options,
                       const TVMTensorShapes& input_shapes);

  virtual void compileTVMModule(const TvmEPOptions& options,
                                const TVMTensorShapes& input_shapes) = 0;
 protected:
  ModulePtr mod_;
};

class TVMCompiler : public TVMCompilerBase {
 public:
  TVMCompiler() = delete;
  ~TVMCompiler() = default;

  TVMCompiler(std::string&& onnx_model_str,
              const std::string& model_path,
              int opset);

  void compileTVMModule(const TvmEPOptions& options,
                        const TVMTensorShapes& input_shapes) final;

 private:
  std::string onnx_model_str_;
  std::string model_path_;
  int opset_;
};

class TVMSoCompiler : public TVMCompilerBase {
 public:
  TVMSoCompiler() = default;
  ~TVMSoCompiler() = default;

  void compileTVMModule(const TvmEPOptions& options,
                        const TVMTensorShapes& input_shapes) final;
};

}   // namespace tvm
}   // namespace onnxruntime

#endif  // TVM_COMPILER_H
