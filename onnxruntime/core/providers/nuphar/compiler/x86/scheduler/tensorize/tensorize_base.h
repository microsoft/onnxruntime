// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <tvm/tvm.h>

// TODO move this tvm_codegen

namespace onnxruntime {
namespace tvm_codegen {

#define TENSORIZE_CLASS(tensorize_name)                                         \
  class tensorize_name : public tvm_codegen::TensorizeBase {                    \
   public:                                                                      \
    tensorize_name(const std::string& name, const std::vector<int32_t>& shape); \
    virtual ~tensorize_name() = default;                                        \
    tvm::TensorIntrin CreateTensorIntrin() override;                            \
  };

#define TENSORIZE_CLASS_WITH_LLVM_IMPORT(tensorize_name)                        \
  class tensorize_name : public tvm_codegen::TensorizeWithLLVMImport {          \
   public:                                                                      \
    tensorize_name(const std::string& name, const std::vector<int32_t>& shape); \
    virtual ~tensorize_name() = default;                                        \
    tvm::TensorIntrin CreateTensorIntrin() override;                            \
    const std::string LLVMImportDef() override;                                 \
  };

// TensorizeBase is for standard tesorization scheduling interface
// For tesorization for nonstandard scheduling, please use custom scheduling
// Name for debug
// Parameter for extra parameter name in Dictionary
// Shape is for the tensorization shape
// CreateTensorIntrin returns tvm::TensorIntrin
class TensorizeBase {
 public:
  TensorizeBase(const std::string& name,
                const std::string& parameter,
                const std::vector<int32_t>& shape)
      : name_(name), parameter_(parameter), shape_(shape) {}

  virtual ~TensorizeBase() = default;

  const std::string& Name() const { return name_; }
  const std::string& Parameter() const { return parameter_; }
  const std::vector<int32_t>& Shape() const { return shape_; }
  const size_t Size() const { return shape_.size(); }

  virtual tvm::TensorIntrin CreateTensorIntrin() = 0;

 protected:
  // name for this tensorization
  std::string name_;
  // specific parameter name
  std::string parameter_;
  // specific shape for this tensorization
  std::vector<int32_t> shape_;
};

// TensorizeWithLLVMImport is also for standard tesorization scheduling interface
// For tesorization for nonstandard scheduling, please use custom scheduling
// LLVMImportDef returns a string of llvm IR
class TensorizeWithLLVMImport : public TensorizeBase {
 public:
  TensorizeWithLLVMImport(const std::string& name,
                          const std::string& parameter,
                          const std::vector<int32_t>& shape)
      : TensorizeBase(name, parameter, shape) {}

  virtual ~TensorizeWithLLVMImport() = default;
  // llvm Func definition
  virtual const std::string LLVMImportDef() = 0;
};

}  // namespace tvm_codegen
}  // namespace onnxruntime
