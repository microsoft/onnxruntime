// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/nuphar_execution_provider.h"

#include "core/codegen/target/ort_tvm_utils.h"  // TODO remove this after removing tvm::runtime
#include "core/common/cpuid_info.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/nuphar/common/analysis/graph_partition_stats.h"
#include "core/providers/nuphar/common/analysis/shape_expr.h"  // TODO: remove this shape_expr after shape_infernece refinement
#include "core/providers/nuphar/common/nuphar_settings.h"
#include "core/providers/nuphar/compiler/x86/x86_target_info.h"
#include "core/providers/nuphar/partition/fuse_rules/node_use_count.h"
#include "core/providers/nuphar/partition/fuse/fuse.h"

#include <tvm/runtime/device_api.h>  // TODO remove this after removing tvm::runtime

using namespace onnxruntime::nuphar;

// from onnxruntime_typeinf.cc, in global namespace
const onnxruntime::DataTypeImpl* ElementTypeFromProto(int type);

namespace onnxruntime {

thread_local std::unique_ptr<std::unordered_map<std::string, int64_t>> NupharExecutionProvider::tls_realized_dims_;

NupharExecutionProvider::NupharExecutionProvider(const NupharExecutionProviderInfo& info)
    : IExecutionProvider(kNupharExecutionProvider) {
  // CodeGenSettings
  nuphar_codegen::CreateNupharCodeGenSettings();

  std::string target_str = info.target_str;
  if (target_str == "")
    target_str = default_nuphar_target_str;
  if (target_str == "llvm") {
    codegen::CodeGenSettings& settings = codegen::CodeGenSettings::Instance();
    if (settings.HasOption(nuphar_codegen::kNupharCodeGenTarget)) {
      if (settings.OptionMatches(nuphar_codegen::kNupharCodeGenTarget, "avx2"))
        codegen_target_ = CodeGenTarget_AVX2();
      else if (settings.OptionMatches(nuphar_codegen::kNupharCodeGenTarget, "avx512"))
        codegen_target_ = CodeGenTarget_AVX512();
      else
        ORT_ENFORCE(false, "Target except avx2/avx512 are not supported!");
    } else {
      const auto& cpu_id_info = CPUIDInfo::GetCPUIDInfo();
      if (cpu_id_info.HasAVX512f())
        codegen_target_ = CodeGenTarget_AVX512();
      else if (cpu_id_info.HasAVX2())
        codegen_target_ = CodeGenTarget_AVX2();
      else
        codegen_target_ = std::make_unique<CodeGenTargetX86>(target_str, 128, 1);  // TODO: use real values
    }
  } else {
    codegen_target_ = std::make_unique<CodeGenTarget>(target_str);
  }

  CreateTVMTarget();

  tvm_host_target_ = tvm::Target();
  tvm_ctx_.device_type = static_cast<DLDeviceType>(tvm_target_->device_type);
  tvm_ctx_.device_id = info.device_id;

  graph_stats_ = std::make_unique<codegen::GraphPartitionStats>();

  whole_graph_shape_infer_ = std::make_shared<ShapeExprContext>();

  RegisterFuseRules();

  DeviceAllocatorRegistrationInfo allocator_info(
      {OrtMemTypeDefault,
       [this](int /*id*/) { return std::make_unique<NupharAllocator>(this->tvm_ctx_); },
       std::numeric_limits<size_t>::max()});

  InsertAllocator(CreateAllocator(allocator_info, tvm_ctx_.device_id));

  // TODO add multi-target support
  tvm_codegen_manager_ = std::make_unique<tvm_codegen::TVMCodeGenManager>();

  // Create codegen handle for one target for now
  // TODO add multi-target support
  nuphar_codegen_handles_.clear();
  auto handle = std::make_unique<tvm_codegen::NupharCodeGenHandle>();
  tvm_codegen_manager_->Initialization();
  tvm_codegen_manager_->SetCodeGenHandle(handle.get());
  handle->allocator = GetAllocator(0, OrtMemTypeDefault);
  handle->codegen_target = codegen_target_.get();
  handle->domain_version_lookup_func =
      [this](const std::string& domain) {
        return GetDomainVersion(domain);
      };

  handle->shape_inference = whole_graph_shape_infer_;

  // TODO: remove
  handle->enable_per_node_parallelized = info.enable_per_node_parallel;
  // TODO: remove
  handle->allow_unaligned_buffers = info.allow_unaligned_buffers;  // TODO remove this

  nuphar_codegen_handles_.push_back(std::move(handle));

  // Runtime Handle
  runtime_handle_ = std::make_unique<nuphar::NupharRuntimeHandle>(tvm_ctx_, info.allow_unaligned_buffers);
  runtime_handle_->allocator = GetAllocator(0, OrtMemTypeDefault);
}

Status NupharExecutionProvider::CopyTensor(const Tensor& src, Tensor& dst) const {
  if (!(strcmp(src.Location().name, TVM_STACKVM) == 0 && strcmp(dst.Location().name, TVM_STACKVM) == 0) &&
      !(strcmp(src.Location().name, TVM_STACKVM) == 0 && strcmp(dst.Location().name, CPU) == 0) &&
      !(strcmp(src.Location().name, CPU) == 0 && strcmp(dst.Location().name, TVM_STACKVM) == 0))
    ORT_NOT_IMPLEMENTED("copy to ", dst.Location().name, " from ", src.Location().name, " is not implemented");

  size_t bytes = src.DataType()->Size() * src.Shape().Size();
  const void* src_data = src.DataRaw();
  void* dst_data = dst.MutableDataRaw();

  // TODO change this to some ort api
  tvm::runtime::DeviceAPI::Get(tvm_ctx_)->CopyDataFromTo(
      src_data, /*src_byte_offset*/ 0, dst_data, /*dst_byte_offset*/ 0, bytes, /*src_ctx*/ tvm_ctx_,
      /*dst_ctx*/ tvm_ctx_, /*data_type*/ tvm_codegen::ToTvmDLDataType(src.DataType()), /*stream*/ nullptr);

  return Status::OK();
}

void NupharExecutionProvider::RegisterFuseRules() {
  fuse_rules_.push_back(std::make_unique<RuleNodeUseCount>(graph_stats_.get()));
}

void NupharExecutionProvider::CreateTVMTarget() {
  tvm_target_ = tvm::Target::create(codegen_target_->GetTargetName());
}

std::vector<std::unique_ptr<ComputeCapability>>
NupharExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                       const std::vector<const KernelRegistry*>&) const {
  // Perform shape inference. If shape inference failed,
  // do not run the model through Nuphar
  if (!ShapeInference(graph_viewer, *whole_graph_shape_infer_).IsOK()) {
    LOGS_DEFAULT(WARNING) << "Model shape inference failed, execution won't use nuphar provider.";
    return {};
  }

  // check if all nodes have shape for outputs
  for (const auto& node : graph_viewer.Nodes()) {
    auto s =
        node.ForEachWithIndex(
            node.OutputDefs(),
            [&](const NodeArg& def, size_t index) {
              if (def.Shape())
                return Status::OK();
              else
                return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Node: ", node.Name(),
                                       " has no output shape for ", def.Name());
            });
    if (!s.IsOK()) {
      LOGS_DEFAULT(WARNING) << "Model shape inference incomplete, execution won't use nuphar provider.";
      LOGS_DEFAULT(WARNING) << s.ErrorMessage();
      return {};
    }
  }

  for (const auto& domain_version : graph_viewer.DomainToVersionMap()) {
    auto iter = domain_versions_.find(domain_version.first);
    if (iter == domain_versions_.end())
      domain_versions_.emplace(domain_version.first, domain_version.second);
    else {
      ORT_ENFORCE(iter->second == domain_version.second,
                  "Inconsistent domain_to_opset_map in Nuphar provider. "
                  "Please create one Nuphar provider instance for each session.");
    }
  }
  // construct graph
  codegen::Promote<codegen::GraphPartitionStats>(graph_stats_)->SetShapeInference(whole_graph_shape_infer_);
  graph_stats_->Evaluate(graph_viewer);

  std::set<NodeIndex> nodes_indexes;
  for (auto& node : graph_viewer.Nodes()) {
    nodes_indexes.insert(node.Index());
  }

  // TODO: currently, we only handle a single fuse rule. Need to change it later.
  ORT_ENFORCE(fuse_rules_.size() == 1);

  std::set<NodeIndex> claimed_nodes;
  // Since we only support a single fuse rule at the moment, we simply passed the rule's Fuse function
  // the "result" that is our final result returned back to the Lotus runtime. It's very likely later we
  // will change the Fuse interface to not take a std::vector<std::unique_ptr<ComputeCapability>> argument.
  std::vector<std::unique_ptr<ComputeCapability>> result;
  ORT_ENFORCE(fuse_rules_[0]->Fuse(
                                graph_viewer,
                                [this](const Node& node) {
                                  return GetKernelRegistry()->TryFindKernel(node, Type()) != nullptr;
                                },
                                claimed_nodes,
                                result)
                  .IsOK());

  // for any node being fused, save initializer tensors of each claimed_nodes
  // because IExecutionProvider::Compile would be called without OpKernelInfo
  const auto& all_initializers = graph_viewer.GetAllInitializedTensors();
  for (auto node_idx : claimed_nodes) {
    const auto& node = graph_viewer.GetNode(node_idx);
    node->ForEachDef(
        [this, &all_initializers](const NodeArg& def, bool is_input) {
          auto iter = all_initializers.find(def.Name());
          if (iter != all_initializers.end()) {
            ORT_ENFORCE(SaveInitializer(def.Name(), iter->second).IsOK());
          }
        });
  }

  // check if we have any node left
  for (auto node_idx : nodes_indexes) {
    if (claimed_nodes.count(node_idx))
      continue;

    if (GetKernelRegistry()->TryFindKernel(*(graph_viewer.GetNode(node_idx)), Type())) {
      std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();
      sub_graph->nodes.push_back(node_idx);
      result.push_back(std::make_unique<ComputeCapability>(std::move(sub_graph)));
    }
  }

  return result;
}

Status NupharExecutionProvider::SaveInitializer(
    const std::string& name,
    const ONNX_NAMESPACE::TensorProto* proto) const {
  auto iter = initializers_used_in_compiled_nodes_.find(name);
  if (iter == initializers_used_in_compiled_nodes_.end()) {
    // create tensor from TensorProto
    // note that session has not call SaveInitializedTensors yet,
    // so we need to make our own copy
    const auto& dims = proto->dims();
    std::vector<int64_t> shape_dims(dims.size());
    for (int i = 0; i < dims.size(); ++i)
      shape_dims[i] = dims[i];

    const TensorShape& shape = TensorShape::ReinterpretBaseType(shape_dims);
    auto data_type = ElementTypeFromProto(proto->data_type());
    auto t = std::make_unique<Tensor>(
        data_type,
        shape,
        GetAllocator(0, OrtMemTypeDefault)->Alloc(shape.Size() * data_type->Size()),
        GetAllocator(0, OrtMemTypeDefault)->Info());

#define CASE_UNPACK_TENSOR(V, T)                                       \
  case V:                                                              \
    ORT_RETURN_IF_ERROR(utils::UnpackTensor<T>(                        \
        *proto,                                                        \
        proto->raw_data().size() ? proto->raw_data().data() : nullptr, \
        proto->raw_data().size(),                                      \
        t->MutableData<T>(),                                           \
        shape.Size()));                                                \
    break;

    switch (proto->data_type()) {
      CASE_UNPACK_TENSOR(ONNX_NAMESPACE::TensorProto_DataType_BOOL, bool);
      CASE_UNPACK_TENSOR(ONNX_NAMESPACE::TensorProto_DataType_DOUBLE, double);
      CASE_UNPACK_TENSOR(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, float);
      CASE_UNPACK_TENSOR(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, MLFloat16);
      CASE_UNPACK_TENSOR(ONNX_NAMESPACE::TensorProto_DataType_INT8, int8_t);
      CASE_UNPACK_TENSOR(ONNX_NAMESPACE::TensorProto_DataType_INT16, int16_t);
      CASE_UNPACK_TENSOR(ONNX_NAMESPACE::TensorProto_DataType_INT32, int32_t);
      CASE_UNPACK_TENSOR(ONNX_NAMESPACE::TensorProto_DataType_INT64, int64_t);
      CASE_UNPACK_TENSOR(ONNX_NAMESPACE::TensorProto_DataType_UINT8, uint8_t);
      CASE_UNPACK_TENSOR(ONNX_NAMESPACE::TensorProto_DataType_UINT16, uint16_t);
      CASE_UNPACK_TENSOR(ONNX_NAMESPACE::TensorProto_DataType_UINT32, uint32_t);
      CASE_UNPACK_TENSOR(ONNX_NAMESPACE::TensorProto_DataType_UINT64, uint64_t);
      default:
        ORT_NOT_IMPLEMENTED("Unimplemented type: ", proto->data_type());
    }

    initializers_used_in_compiled_nodes_.emplace(
        name,
        std::move(t));
  }
  return Status::OK();
}

// Compile nodes into node_compute_funcs
Status NupharExecutionProvider::Compile(
    const std::vector<onnxruntime::Node*>& nodes,
    std::vector<NodeComputeInfo>& node_compute_funcs) {
  // save a copy of fused_nodes pointers to make sure lambda capture is valid
  size_t num_existing_nodes = compiled_nodes_.size();
  compiled_nodes_.insert(compiled_nodes_.end(), nodes.begin(), nodes.end());

  for (size_t i = num_existing_nodes; i < compiled_nodes_.size(); ++i) {
    const auto& node = compiled_nodes_[i];
    NodeComputeInfo info;

    // Create state function
    // This is similar to the original OpKernel constructor
    // TODO move compilation part out of create_state_func to above
    info.create_state_func =
        [&](ComputeContext* ctx, FunctionState* state) {
          // TODO: remove unique_ptr
          std::unique_ptr<NupharFunctionState> s =
              std::make_unique<NupharFunctionState>(
                  *node,
                  [this](const std::string& name, const Tensor** tensor) {
                    auto iter = initializers_used_in_compiled_nodes_.find(name);
                    if (iter == initializers_used_in_compiled_nodes_.end()) {
                      *tensor = nullptr;
                      return false;
                    } else {
                      *tensor = iter->second.get();
                      return true;
                    }
                  },
                  *ctx,
                  this);

          *state = s.release();
          return 0;
        };

    // Release state function
    // This is similar to the original OpKernel destructor
    info.release_state_func =
        [](FunctionState state) {
          if (state)
            delete static_cast<NupharFunctionState*>(state);
        };

    // Compute function
    // This is similar to the original OpKernel's Compute()
    info.compute_func =
        [](FunctionState state, const OrtCustomOpApi*, OrtKernelContext* op_kernel_context) {
          NupharFunctionState* s = reinterpret_cast<NupharFunctionState*>(state);
          s->Compute(reinterpret_cast<OpKernelContext*>(op_kernel_context));
          return 0;
        };

    node_compute_funcs.push_back(info);
  }
  return Status::OK();
}

#define NUPHAR_OP(name, ver, types) \
  class ONNX_OPERATOR_KERNEL_CLASS_NAME(kNupharExecutionProvider, kOnnxDomain, ver, name);

#define NUPHAR_VERSIONED_OP(name, start_ver, end_ver, types) \
  class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kNupharExecutionProvider, kOnnxDomain, start_ver, end_ver, name);

LIST_NUPHAR_OPS()

#undef NUPHAR_OP
#undef NUPHAR_VERSIONED_OP

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kNupharExecutionProvider, kOnnxDomain, 6, 8, Cast);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kNupharExecutionProvider, kOnnxDomain, 9, Cast);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kNupharExecutionProvider, kOnnxDomain, 1, Gather);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kNupharExecutionProvider, kOnnxDomain, 10, MatMulInteger);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kNupharExecutionProvider, kOnnxDomain, 9, Scan);

static void RegisterStandaloneNupharKernels(KernelRegistry& kernel_registry) {
#define NUPHAR_OP(name, ver, types) \
  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kNupharExecutionProvider, kOnnxDomain, ver, name)>());

#define NUPHAR_VERSIONED_OP(name, start_ver, end_ver, types) \
  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kNupharExecutionProvider, kOnnxDomain, start_ver, end_ver, name)>());

  LIST_NUPHAR_OPS()

#undef NUPHAR_OP
#undef NUPHAR_VERSIONED_OP

  // ops that have multiple type constraints
  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kNupharExecutionProvider, kOnnxDomain, 6, 8, Cast)>());
  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kNupharExecutionProvider, kOnnxDomain, 9, Cast)>());
  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kNupharExecutionProvider, kOnnxDomain, 1, Gather)>());
  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kNupharExecutionProvider, kOnnxDomain, 10, MatMulInteger)>());
  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kNupharExecutionProvider, kOnnxDomain, 9, Scan)>());
}

std::shared_ptr<KernelRegistry> NupharExecutionProvider::GetKernelRegistry() const {
  if (kernel_registry_ == nullptr) {
    kernel_registry_ = std::make_shared<KernelRegistry>();
    RegisterStandaloneNupharKernels(*kernel_registry_);
  }
  return kernel_registry_;
}

}  // namespace onnxruntime
