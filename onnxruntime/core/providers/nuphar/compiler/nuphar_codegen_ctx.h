// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// TODO: clean header

#include <tvm/tvm.h>
#include "core/codegen/target/tvm_context.h"
#include "core/codegen/common/common.h"
#include "core/providers/nuphar/compiler/nuphar_handle.h"
#include "core/providers/nuphar/compiler/tvm_initializer.h"
#include "core/common/common.h"
#include "core/graph/graph.h"
#include "core/providers/nuphar/common/analysis/graph_stats.h"

namespace onnxruntime {
namespace tvm_codegen {

// Nuphar Tensor Context
struct TVMTensorCtx {
  std::map<std::string, tvm::Tensor> inputs;
  std::map<const Node*, tvm::Array<tvm::Tensor>> ops;
  std::map<std::string, std::pair<const Node*, size_t>> input_from;
  std::map<const Node*, std::vector<std::pair<tvm::Tensor, tvm::Tensor>>> loop_states;

  std::vector<std::pair<tvm::Tensor, tvm::Tensor>>&
  LookupLoopStates(const Node* node) {
    auto iter = loop_states.find(node);
    ORT_ENFORCE(iter != loop_states.end());
    return iter->second;
  }

  const tvm::Tensor
  Lookup(const NodeArg* def) const {
    const std::string& def_name = def->Name();
    auto iter = inputs.find(def_name);
    if (iter != inputs.end())
      return iter->second;

    auto iter_out_index = input_from.find(def_name);

    // OK if shape inference is incomplete
    // This is for some per-node unit test where NodeArg does not even have shape ranks
    // We ignore the shape inference in ToCapacity computation in per-node unit tests
    if (iter_out_index == input_from.end())
      return tvm::Tensor();  // TODO confirm this

    const Node* from_node = iter_out_index->second.first;
    size_t index = iter_out_index->second.second;
    auto iter_op = ops.find(from_node);
    ORT_ENFORCE(iter_op != ops.end());
    return iter_op->second[index];
  }
};

struct WeightLayoutCtx {
  //std::unordered_map<std::string, std::string> initializer_to_weight_layout;  // unused yet. This is for decoupling weight layout compile and run
  std::unordered_map<std::string, tvm::runtime::PackedFunc> weight_layout_to_packed_func;
};

// TODO: gradually move CodeGenContext's members to NupharCodeGenCtx
//       until CodeGenContext becoming generic
class NupharCodeGenCtx : public CodeGenContext {
 public:
  NupharCodeGenCtx(const Node& node,
                   InitializerMap& initializers,
                   const NupharCodeGenHandle* handle);

  virtual ~NupharCodeGenCtx() = default;

  const codegen::OrtGraphStats* GetGraphStats() const;

  InitializerInfo* GetInitializerInfo(const std::string& name);
  const InitializerInfo* GetInitializerInfo(const std::string& name) const;
  bool IsInitializerMarshalled(const std::string& name) const;
  const InitializerMap& GetInitializerMap() const;
  // TODO: remove this
  size_t SizeInitializerMarshalled() const;

  // On-the-fly apply an existing layout
  tvm::Tensor ApplyWeightLayout(
      const std::string& layout_key,
      const std::string& initializer_name,
      const tvm::Tensor& X,
      bool returnMarshalled);

  void RecordTensorToNode(const tvm::Tensor& t, const Node* node);
  const Node* FindNode(const tvm::Tensor& t) const;
  const Tensor* GetOrtInitializedTensor(const NodeArg* def) const;

  const NupharCodeGenHandle* GetCodeGenHandle() const;

  template <typename T>
  IAllocatorUniquePtr<T> AllocateT(size_t size) const { return IAllocator::MakeUniquePtr<T>(nuphar_handle_->allocator, size); }

  IAllocatorUniquePtr<void> Allocate(size_t size) const { return AllocateT<void>(size); }

  // Keep for CodeGenContext
  TVMTensorCtx& GetTVMTensorCtx() {
    return tvm_tensor_ctx_;
  }

  // Keep for CodeGenContext
  const TVMTensorCtx& GetTVMTensorCtx() const {
    return tvm_tensor_ctx_;
  }

 private:
  std::unique_ptr<codegen::OrtGraphStats> graph_stats_;

  // initializer lookup table
  InitializerMap& initializer_map_;

  const NupharCodeGenHandle* nuphar_handle_;

  // A table from tvm::Tensor (its unchanged source tvm::Node*) to ORT Node
  std::unordered_map<const tvm::Node*, const Node*> tvm_tensor_to_node_lookup_;

  // All TVM Tensor and correponidng shape context
  TVMTensorCtx tvm_tensor_ctx_;

  // all layout
  WeightLayoutCtx weight_layout_ctx_;
};

}  // namespace tvm_codegen
}  // namespace onnxruntime
