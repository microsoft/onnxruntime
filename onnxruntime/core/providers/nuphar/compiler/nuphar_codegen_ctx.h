// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/codegen/common/common.h"
#include "core/codegen/passes/utils/codegen_context.h"
#include "core/common/common.h"
#include "core/graph/graph.h"
#include "core/providers/nuphar/common/analysis/graph_stats.h"
#include "core/providers/nuphar/common/nuphar_subgraph.h"
#include "core/providers/nuphar/compiler/initializer_info.h"
#include "core/providers/nuphar/compiler/nuphar_handle.h"

#include <set>
#include <tvm/tvm.h>

namespace onnxruntime {
namespace nuphar {

// Nuphar Tensor Context
struct TVMTensorCtx {
  std::map<std::string, tvm::Tensor> inputs;
  std::map<const Node*, tvm::Array<tvm::Tensor>> ops;
  std::map<std::string, std::pair<const Node*, size_t>> input_from;

  bool Lookup(const NodeArg* def, tvm::Tensor& tensor) {
    const std::string& def_name = def->Name();
    auto iter = inputs.find(def_name);
    if (iter != inputs.end()) {
      tensor = iter->second;
      return true;
    }

    auto iter_out_index = input_from.find(def_name);

    if (iter_out_index == input_from.end()) {
      return false;
    }

    const Node* from_node = iter_out_index->second.first;
    size_t index = iter_out_index->second.second;
    auto iter_op = ops.find(from_node);
    ORT_ENFORCE(iter_op != ops.end());
    tensor = iter_op->second[index];
    return true;
  }

  const tvm::Tensor
  Lookup(const NodeArg* def) const {
    const std::string& def_name = def->Name();
    auto iter = inputs.find(def_name);
    if (iter != inputs.end()) {
      return iter->second;
    }

    auto iter_out_index = input_from.find(def_name);

    ORT_ENFORCE(iter_out_index != input_from.end());

    const Node* from_node = iter_out_index->second.first;
    size_t index = iter_out_index->second.second;
    auto iter_op = ops.find(from_node);
    ORT_ENFORCE(iter_op != ops.end());
    return iter_op->second[index];
  }
};

struct WeightLayoutCtx {
  //std::map<std::string, std::string> initializer_to_weight_layout;  // unused yet. This is for decoupling weight layout compile and run
  std::unordered_map<std::string, tvm::runtime::PackedFunc> weight_layout_to_packed_func;
};

// NupharCodeGenCtx is Nuphar-specific CodeGenContext
class NupharCodeGenCtx : public tvm_codegen::CodeGenContext {
 public:
  NupharCodeGenCtx(const Node& node,
                   const std::map<std::string, const Tensor*>& initializers,
                   std::unordered_map<std::string, std::unique_ptr<Tensor>>& global_generated_initializers,
                   const NupharCodeGenHandle* handle);

  NupharCodeGenCtx(const nuphar::NupharSubgraphUnit& subgraph,
                   std::unordered_map<std::string, std::unique_ptr<Tensor>>& global_generated_initializers,
                   const NupharCodeGenHandle* handle);

  virtual ~NupharCodeGenCtx() = default;

  const NupharSubgraphUnitStats* GetGraphStats() const;

  bool IsInitializer(const std::string& name) const;
  const Tensor* GetOrtInitializerTensor(const std::string& name) const;
  WeightLayoutCodegenInfo* GetWeightLayoutInfo(const std::string& name);
  const WeightLayoutCodegenInfo* GetWeightLayoutInfo(const std::string& name) const;
  void CreateWeightLayoutInfo(const std::string& name, const tvm::Tensor& tensor);
  const std::map<std::string, std::unique_ptr<WeightLayoutCodegenInfo>>& GetWeightLayoutMap() const;

  // On-the-fly apply an existing layout
  tvm::Tensor ApplyWeightLayout(
      const std::string& layout_key,
      const std::string& initializer_name,
      const tvm::Tensor& X,
      bool returnMarshalled);

  void RecordTensorToNode(const tvm::Tensor& t, const Node* node);
  const Node* FindNode(const tvm::Tensor& t) const;

  const NupharCodeGenHandle* GetCodeGenHandle() const;

  // TODO remove this after decoupling compiler and runtime of WeightLayout
  template <typename T>
  IAllocatorUniquePtr<T> AllocateT(size_t size) const { return IAllocator::MakeUniquePtr<T>(nuphar_handle_->allocator, size); }
  // TODO remove this after decoupling compiler and runtime of WeightLayout
  IAllocatorUniquePtr<void> Allocate(size_t size) const { return AllocateT<void>(size); }

  // Keep for CodeGenContext
  TVMTensorCtx& GetTVMTensorCtx() {
    return tvm_tensor_ctx_;
  }

  // Keep for CodeGenContext
  const TVMTensorCtx& GetTVMTensorCtx() const {
    return tvm_tensor_ctx_;
  }

  void InsertLiteral(const std::string& str) {
    literalized_scalars_.insert(str);
  }

  bool CheckLiteral(const std::string& str) {
    return literalized_scalars_.count(str) > 0;
  }

 private:
  std::set<std::string> literalized_scalars_;

  std::unique_ptr<NupharSubgraphUnitStats> graph_stats_;

  const NupharCodeGenHandle* nuphar_handle_;

  const std::map<std::string, const Tensor*>& initializers_;

  // A table from tvm::Tensor (its unchanged source tvm::Node*) to ORT Node
  std::unordered_map<const tvm::Node*, const Node*> tvm_tensor_to_node_lookup_;

  // All TVM Tensor and correponidng shape context
  TVMTensorCtx tvm_tensor_ctx_;

  // local copy
  std::map<std::string, std::unique_ptr<WeightLayoutCodegenInfo>> initializer_layouts_;

  std::unordered_map<std::string, std::unique_ptr<Tensor>>& global_generated_initializers_;

  // all layouts
  WeightLayoutCtx weight_layout_ctx_;
};

}  // namespace nuphar
}  // namespace onnxruntime
