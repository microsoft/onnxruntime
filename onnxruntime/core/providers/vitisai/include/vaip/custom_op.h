// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "./dll_safe.h"
#include "./my_ort.h"

struct OrtApi;
typedef struct OrtApi OrtApi;
typedef struct OrtKernelContext OrtKernelContext;
struct OrtCustomOpDomain;
typedef struct OrtCustomOpDomain OrtCustomOpDomain;

namespace vaip_core {
class PassContext;
class MetaDefProto;
class CustomOp;
class ExecutionProvider {
 public:
  VAIP_DLL_SPEC ExecutionProvider();
  virtual ~ExecutionProvider();

 public:
  virtual DllSafe<std::vector<std::string>> get_meta_def_inputs() const = 0;
  virtual DllSafe<std::vector<std::string>> get_meta_def_outputs() const = 0;
  virtual DllSafe<std::vector<std::string>> get_meta_def_nodes() const = 0;
  virtual DllSafe<std::vector<std::string>>
  get_meta_def_constant_initializer() const = 0;
  virtual bool get_meta_def_fallback_CPU() const { return false; };
  virtual std::unique_ptr<CustomOp> compile() const = 0;

 public:
  inline void set_fused_node(const onnxruntime::Node* fused_node) { fused_node_ = fused_node; }
  inline const onnxruntime::Node* get_fused_node() const { return fused_node_; }
  inline void set_model(onnxruntime::Model* model) { model_ = model; }
  inline onnxruntime::Model* get_model() const { return model_; }

 private:
  const onnxruntime::Node* fused_node_ = nullptr;
  onnxruntime::Model* model_ = nullptr;
};

class CustomOp {
 public:
  VAIP_DLL_SPEC CustomOp();
  VAIP_DLL_SPEC virtual ~CustomOp();

 public:
  virtual void Compute(const OrtApi* api, OrtKernelContext* context) const = 0;
};

}  // namespace vaip_core
