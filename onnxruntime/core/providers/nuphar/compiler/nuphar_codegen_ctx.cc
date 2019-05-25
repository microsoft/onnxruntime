// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "nuphar_codegen_ctx.h"

#include "core/codegen/common/common.h"
#include "core/codegen/common/utils.h"
#include "core/codegen/mti/mti_tvm_utils.h"  // TODO: remove this after decoupling layout compile and run
#include "core/providers/nuphar/common/analysis/subgraph_gen_stats.h"
#include "core/codegen/target/ort_tvm_utils.h"  // TODO: remove this after decoupling layout compile and run
#include <tvm/build_module.h>                   // TODO: remove this after decoupling layout compile and run

#include "core/providers/nuphar/common/nuphar_tvm_utils.h"

namespace onnxruntime {
namespace tvm_codegen {

NupharCodeGenCtx::NupharCodeGenCtx(
    const Node& node,
    InitializerMap& initializers,
    const NupharCodeGenHandle* handle)
    : CodeGenContext(handle),
      nuphar_handle_(handle),
      initializer_map_(initializers) {
  graph_stats_ = std::make_unique<codegen::SubGraphStats>(nuphar_handle_->shape_inference);
  const Graph* subgraph = GetSubgraph(node);
  if (nullptr != subgraph && !IsFusedNode(node)) {
    codegen::Promote<codegen::SubGraphStats>(graph_stats_)
        ->Evaluate(GraphViewer(*subgraph));
  } else {
    codegen::Promote<codegen::SubGraphStats>(graph_stats_)
        ->EvaluateSingleNode(node);
  }
}

// This is a temp function before we decouple weight layout compilation and run
// OLD code
// This will be moved.
// TODO: remove this.
static tvm::runtime::PackedFunc LowerLayoutFunc(const WeightLayout* layout) {
  tvm::Array<tvm::Tensor> inputs;
  tvm::Array<tvm::Tensor> outputs;

  layout->CreateLayoutMarshallingTVMOp(inputs, outputs);

  auto config = tvm::build_config();
  config->disable_select_rewriting = true;
  auto S = tvm::create_schedule({outputs[0]->op});
  S[outputs[0]->op].compute_root();

  std::string func_name = layout->Name() + "_marshall";

  tvm::runtime::PackedFunc cached_func = nuphar_codegen::LoadTVMPackedFuncFromCache(func_name);

  if (cached_func == nullptr) {
    auto lowered = tvm::lower(S, {inputs[0], outputs[0]}, func_name, {}, config);
    auto module = tvm::build(lowered, tvm::target::llvm(), tvm::Target(), config);
    DumpTVMModuleToFile(func_name, module);
    nuphar_codegen::SaveTVMModuleToCache(func_name, module);
    cached_func = module.GetFunction(func_name);
  }
  return cached_func;
}

// This is a temp function before we decouple weight layout compilation and run.
// This will be moved.
// TODO: remove this.
static std::shared_ptr<Tensor> Marshalling(
    const std::string& initializer_name,
    const Tensor* original_initializer,
    const WeightLayout* layout_ptr,
    WeightLayoutCtx& ctx_layout,
    AllocatorPtr allocator) {
  tvm::runtime::PackedFunc packed_func;

  const std::string& layout_key = layout_ptr->Name();
  if (ctx_layout.weight_layout_to_packed_func.count(layout_key) == 0) {
    packed_func = LowerLayoutFunc(layout_ptr);
    ctx_layout.weight_layout_to_packed_func.insert(std::make_pair(layout_key, packed_func));
  } else {
    packed_func = ctx_layout.weight_layout_to_packed_func[layout_key];
  }

  std::vector<int64_t> marshalled_shape = layout_ptr->ToActualShape(original_initializer);
  auto marshalled_size = TotalSize(marshalled_shape);
  auto byte_size = original_initializer->DataType()->Size();

  std::shared_ptr<Tensor> out;
  void* p_data = allocator->Alloc(marshalled_size * byte_size);
  out = std::make_shared<Tensor>(
      original_initializer->DataType(),
      TensorShape(marshalled_shape),
      p_data,
      allocator->Info());

  int num_args = 2;
  DLContext tvm_ctx{kDLCPU, 0};
  std::vector<TVMValue> lvalues(num_args);
  std::vector<DLTensor> tvm_tensors(num_args);

  // input
  const auto& tensor_shape = original_initializer->Shape();
  auto input_shape = tensor_shape.GetDims();
  if (input_shape.empty())
    input_shape.push_back(1);
  const void* input_data = original_initializer->DataRaw();
  DLDataType tvm_dtype = tvm_codegen::ToTvmDLDataType(original_initializer->DataType());

  tvm_tensors[0] = {const_cast<void*>(input_data), tvm_ctx,
                    gsl::narrow_cast<int>(input_shape.size()), tvm_dtype,
                    input_shape.data(), nullptr, 0};
  lvalues[0].v_handle = &(tvm_tensors[0]);

  // output
  tvm_tensors[1] = {p_data, tvm_ctx,
                    gsl::narrow_cast<int>(marshalled_shape.size()), tvm_dtype,
                    marshalled_shape.data(), nullptr, 0};
  lvalues[1].v_handle = &(tvm_tensors[1]);

  auto types_code = std::vector<int>(num_args, kNDArrayContainer);
  tvm::TVMArgs tvm_args(lvalues.data(), types_code.data(), num_args);
  tvm::TVMRetValue rvalue;
  packed_func.CallPacked(tvm_args, &rvalue);
  return out;
}

// on the fly WeightLayout transformer
tvm::Tensor NupharCodeGenCtx::ApplyWeightLayout(
    const std::string& layout_key,
    const std::string& initializer_name,
    const tvm::Tensor& X,
    bool returnMarshalled) {
  tvm::Tensor marshalled;
  auto info = GetInitializerInfo(initializer_name);
  ORT_ENFORCE(nullptr != info);
  auto layout_info = info->layout_info.get();
  ORT_ENFORCE(nullptr != layout_info);

  auto layout_ptr = nuphar_handle_->layout_registry->Get(layout_key);
  ORT_ENFORCE(nullptr != layout_ptr);

  // check whether the weight is applied layout marshalling
  if (nullptr == layout_info->marshalled_initializer) {
    ORT_ENFORCE(!layout_info->is_marshalled);  // initializer should not have been marshalled before

    // TODO: change to delayed call
    layout_info->layout = layout_ptr->Name();

    // TODO: change to delayed call
    layout_info->marshalled_initializer =
        Marshalling(initializer_name,
                    info->original_initializer,
                    layout_ptr,
                    weight_layout_ctx_,
                    nuphar_handle_->allocator);

    layout_info->marshalled_tensor = tvm::placeholder(layout_ptr->ToActualShape(X), X->dtype, initializer_name + "_marshalled");
    layout_info->unmarshalled_tensor = tvm::compute(
        X->shape,
        [&](const tvm::Array<tvm::Var>& nominal_coord) {
          tvm::Array<tvm::Expr> cc;
          for (auto v : nominal_coord)
            cc.push_back(v);

          auto coord_trans_func = layout_ptr->ToActual(X);
          return layout_info->marshalled_tensor(coord_trans_func(cc));
        },
        initializer_name + "_unmarshalled");

    layout_info->is_marshalled = true;

  } else {
    ORT_ENFORCE(layout_ptr->Name() == layout_info->layout);
  }

  if (returnMarshalled) {
    return layout_info->marshalled_tensor;
  }
  return layout_info->unmarshalled_tensor;
}

const codegen::OrtGraphStats* NupharCodeGenCtx::GetGraphStats() const {
  return graph_stats_.get();
}

InitializerInfo* NupharCodeGenCtx::GetInitializerInfo(const std::string& name) {
  if (initializer_map_.count(name) > 0)
    return &initializer_map_.at(name);
  else
    return nullptr;
}

const InitializerInfo* NupharCodeGenCtx::GetInitializerInfo(const std::string& name) const {
  if (initializer_map_.count(name) > 0)
    return &initializer_map_.at(name);
  else
    return nullptr;
}

bool NupharCodeGenCtx::IsInitializerMarshalled(const std::string& name) const {
  auto info = GetInitializerInfo(name);
  if (nullptr == info)
    return false;
  return (nullptr != info->layout_info);
}

const InitializerMap& NupharCodeGenCtx::GetInitializerMap() const {
  return initializer_map_;
}

size_t NupharCodeGenCtx::SizeInitializerMarshalled() const {
  size_t count = 0;
  for (const auto& item : initializer_map_) {
    const auto& info = item.second;
    if (nullptr != info.layout_info) {
      ++count;
    }
  }
  return count;
}

void NupharCodeGenCtx::RecordTensorToNode(const tvm::Tensor& t, const Node* node) {
  // Insert tvm::Tensor and Node to the lookup table
  // But bypass it when node is a output alias
  if (!codegen::Promote<codegen::SubGraphStats>(graph_stats_)->IsOutputAlias(node))
    tvm_tensor_to_node_lookup_.insert(std::make_pair(t->op.get(), node));
}

const Node* NupharCodeGenCtx::FindNode(const tvm::Tensor& t) const {
  auto p = tvm_tensor_to_node_lookup_.find(t->op.get());
  if (p != tvm_tensor_to_node_lookup_.end())
    return p->second;
  return nullptr;
}

const Tensor* NupharCodeGenCtx::GetOrtInitializedTensor(const NodeArg* def) const {
  if (nullptr != def) {
    auto iter = initializer_map_.find(def->Name());
    if (iter != initializer_map_.cend())
      return iter->second.original_initializer;
  }
  return nullptr;
}

const NupharCodeGenHandle* NupharCodeGenCtx::GetCodeGenHandle() const {
  return nuphar_handle_;
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
