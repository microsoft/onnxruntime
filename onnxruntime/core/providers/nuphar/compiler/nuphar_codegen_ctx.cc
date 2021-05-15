// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "nuphar_codegen_ctx.h"

#include "core/codegen/common/common.h"
#include "core/codegen/common/utils.h"
#include "core/codegen/mti/mti_tvm_utils.h"  // TODO: remove this after decoupling layout compile and run
#include "core/common/safeint.h"
#include "core/providers/nuphar/common/analysis/subgraph_codegen_stats.h"
#include "core/codegen/passes/utils/ort_tvm_utils.h"  // TODO: remove this after decoupling layout compile and run
#include <tvm/build_module.h>                         // TODO: remove this after decoupling layout compile and run

#include "core/providers/nuphar/common/nuphar_tvm_utils.h"

namespace onnxruntime {
namespace nuphar {

NupharCodeGenCtx::NupharCodeGenCtx(
    const Node& node,
    const std::map<std::string, const Tensor*>& initializers,
    std::unordered_map<std::string, std::unique_ptr<Tensor>>& global_generated_initializers,
    const NupharCodeGenHandle* handle)
    : CodeGenContext(handle),
      nuphar_handle_(handle),
      initializers_(initializers),
      global_generated_initializers_(global_generated_initializers) {
  // construct graph_stats
  graph_stats_ = std::make_unique<CodeGenUnitStats>(nuphar_handle_->shape_inference);
}

NupharCodeGenCtx::NupharCodeGenCtx(
    const nuphar::NupharSubgraphUnit& subgraph,
    std::unordered_map<std::string, std::unique_ptr<Tensor>>& global_generated_initializers,
    const NupharCodeGenHandle* handle)
    : CodeGenContext(handle),
      nuphar_handle_(handle),
      initializers_(subgraph.initializers),
      global_generated_initializers_(global_generated_initializers) {
  graph_stats_ = std::make_unique<CodeGenUnitStats>(nuphar_handle_->shape_inference);
  Promote<CodeGenUnitStats>(graph_stats_)->Evaluate(subgraph);
}

// This is a temp function before we decouple weight layout compilation and run
// This will be moved.
// TODO: remove this.
static tvm::runtime::PackedFunc LowerLayoutFunc(const tvm_codegen::WeightLayout* layout) {
  tvm::Array<tvm::Tensor> inputs;
  tvm::Array<tvm::Tensor> outputs;

  layout->CreateLayoutMarshallingTVMOp(inputs, outputs);

  auto config = tvm::build_config();
  config->disable_select_rewriting = true;
  auto S = tvm::create_schedule({outputs[0]->op});
  S[outputs[0]->op].compute_root();

  std::string func_name = layout->Name() + "_marshall";

  tvm::runtime::PackedFunc cached_func;
  auto cache_status = nuphar::LoadTVMPackedFuncFromCache(func_name, cached_func);
  if (cache_status != nuphar::CacheStatus::Found) {
    ORT_ENFORCE(cached_func == nullptr);
    auto lowered = tvm::lower(S, {inputs[0], outputs[0]}, func_name, {}, config);
    auto module = tvm::build(lowered, tvm::target::llvm(), tvm::Target(), config);
    tvm_codegen::DumpTVMModuleToFile(func_name, module);
    if (cache_status == nuphar::CacheStatus::Missing) {
      nuphar::SaveTVMModuleToCache(func_name, module);
    }
    cached_func = module.GetFunction(func_name);
  }
  return cached_func;
}

// This is a temp function before we decouple weight layout compilation and run.
// This will be moved.
// TODO: remove this.
static const Tensor* Marshalling(
    const std::string& initializer_name,
    std::unordered_map<std::string, std::unique_ptr<Tensor>>& global_generated_initializers,
    const Tensor* original_initializer,
    const tvm_codegen::WeightLayout* layout_ptr,
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

  std::unique_ptr<Tensor> out_ptr;
  void* p_data = allocator->Alloc(SafeInt<size_t>(marshalled_size) * byte_size);
  out_ptr = std::make_unique<Tensor>(
      original_initializer->DataType(),
      TensorShape(marshalled_shape),
      p_data,
      allocator->Info());

  global_generated_initializers.emplace(initializer_name, std::move(out_ptr));

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
  return global_generated_initializers.at(initializer_name).get();
}

// on the fly WeightLayout transformer
tvm::Tensor NupharCodeGenCtx::ApplyWeightLayout(
    const std::string& layout_key,
    const std::string& initializer_name,
    const tvm::Tensor& X,
    bool returnMarshalled) {
  tvm::Tensor marshalled;
  ORT_ENFORCE(IsInitializer(initializer_name));
  auto layout_info = GetWeightLayoutInfo(initializer_name);
  ORT_ENFORCE(nullptr != layout_info);

  const Tensor* original_initializer = GetOrtInitializerTensor(initializer_name);

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
                    global_generated_initializers_,
                    original_initializer,
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

const NupharSubgraphUnitStats* NupharCodeGenCtx::GetGraphStats() const {
  return graph_stats_.get();
}

bool NupharCodeGenCtx::IsInitializer(const std::string& name) const {
  return initializers_.count(name) > 0;
}

const Tensor* NupharCodeGenCtx::GetOrtInitializerTensor(const std::string& name) const {
  if (IsInitializer(name))
    return initializers_.at(name);
  return nullptr;
}

WeightLayoutCodegenInfo* NupharCodeGenCtx::GetWeightLayoutInfo(const std::string& name) {
  if (initializer_layouts_.count(name) > 0)
    return initializer_layouts_.at(name).get();
  return nullptr;
}

const WeightLayoutCodegenInfo* NupharCodeGenCtx::GetWeightLayoutInfo(const std::string& name) const {
  if (initializer_layouts_.count(name) > 0)
    return initializer_layouts_.at(name).get();
  return nullptr;
}

void NupharCodeGenCtx::CreateWeightLayoutInfo(const std::string& name, const tvm::Tensor& tensor) {
  ORT_ENFORCE(initializer_layouts_.count(name) == 0);
  initializer_layouts_.emplace(name, std::move(std::make_unique<WeightLayoutCodegenInfo>(tensor)));
}

const std::map<std::string, std::unique_ptr<WeightLayoutCodegenInfo>>& NupharCodeGenCtx::GetWeightLayoutMap() const {
  return initializer_layouts_;
}

void NupharCodeGenCtx::RecordTensorToNode(const tvm::Tensor& t, const Node* node) {
  // Insert tvm::Tensor and Node to the lookup table
  // But bypass it when node is a output alias
  if (!Promote<CodeGenUnitStats>(graph_stats_)->IsOutputAlias(node))
    tvm_tensor_to_node_lookup_.insert(std::make_pair(t->op.get(), node));
}

const Node* NupharCodeGenCtx::FindNode(const tvm::Tensor& t) const {
  auto p = tvm_tensor_to_node_lookup_.find(t->op.get());
  if (p != tvm_tensor_to_node_lookup_.end())
    return p->second;
  return nullptr;
}

const NupharCodeGenHandle* NupharCodeGenCtx::GetCodeGenHandle() const {
  return nuphar_handle_;
}

}  // namespace nuphar
}  // namespace onnxruntime
