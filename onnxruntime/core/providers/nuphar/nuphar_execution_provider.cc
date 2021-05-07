// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/nuphar_execution_provider.h"

#include "core/codegen/passes/utils/ort_tvm_utils.h"  // TODO remove this after removing tvm::runtime
#include "core/common/cpuid_info.h"
#include "core/common/safeint.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/nuphar/common/analysis/shape_expr.h"  // TODO: remove this shape_expr after shape_infernece refinement
#include "core/providers/nuphar/common/analysis/subgraph_partition_stats.h"
#include "core/providers/nuphar/common/nuphar_settings.h"
#include "core/providers/nuphar/common/utils.h"
#include "core/providers/nuphar/compiler/x86/x86_target_info.h"
#include "core/providers/nuphar/kernel.h"
#include "core/providers/nuphar/partition/graph_partitioner.h"
#include "core/framework/onnxruntime_typeinfo.h"

#include <tvm/runtime/device_api.h>  // TODO remove this after removing tvm::runtime

using namespace onnxruntime::nuphar;

namespace onnxruntime {

// Initialization of thread local counter as subgraph id
thread_local int64_t NupharSubgraphUnit::counter = 0;

thread_local std::unique_ptr<std::unordered_map<std::string, int64_t>> NupharExecutionProvider::tls_realized_dims_;
thread_local int NupharExecutionProvider::per_model_fused_count_ = 0;

static std::string GetCurrentHostTargetString() {
#if USE_TVM_WITH_LLVM
  // auto detect from CPU ID
  const auto& cpu_id_info = CPUIDInfo::GetCPUIDInfo();
  if (cpu_id_info.HasAVX512f()) {
    return CodeGenTargetX86::LLVM_TARGET_AVX512;
  } else if (cpu_id_info.HasAVX2()) {
    return CodeGenTargetX86::LLVM_TARGET_AVX2;
  } else if (cpu_id_info.HasAVX()) {
    return CodeGenTargetX86::LLVM_TARGET_AVX;
  }
  return llvm_target_str;
#else
  return stackvm_target_str;
#endif  // USE_TVM_WITH_LLVM
}

NupharExecutionProvider::NupharExecutionProvider(const NupharExecutionProviderInfo& info)
    : IExecutionProvider(kNupharExecutionProvider) {
  nuphar::CreateNupharCodeGenSettings(info);
  codegen::CodeGenSettings& settings = codegen::CodeGenSettings::Instance();

  std::string target_str;
  if (settings.HasOption(nuphar::kNupharCodeGenTarget)) {
    target_str = settings.GetOptionValue(nuphar::kNupharCodeGenTarget);
  }

  if (target_str.empty()) {
    target_str = default_nuphar_target_str;
  }

  const auto& cpu_id_info = CPUIDInfo::GetCPUIDInfo();
  if (target_str == llvm_target_str) {
    // auto detect from CPU ID
    if (cpu_id_info.HasAVX512f()) {
      codegen_target_ = CodeGenTarget_AVX512();
    } else if (cpu_id_info.HasAVX2()) {
      codegen_target_ = CodeGenTarget_AVX2();
    } else if (cpu_id_info.HasAVX()) {
      codegen_target_ = CodeGenTarget_AVX();
    } else {
      codegen_target_ = std::make_unique<CodeGenTargetX86>(target_str, 128, 1);  // TODO: use real values
    }
  } else if (target_str == "avx") {
    codegen_target_ = CodeGenTarget_AVX();
  } else if (target_str == "avx2") {
    codegen_target_ = CodeGenTarget_AVX2();
  } else if (target_str == "avx512") {
    codegen_target_ = CodeGenTarget_AVX512();
  } else if (target_str != stackvm_target_str) {
    codegen_target_ = std::make_unique<CodeGenTarget>(target_str);
  } else {
    ORT_NOT_IMPLEMENTED("Not supported target, should be one of stackvm/llvm/avx/avx2/avx512.");
  }

  if (settings.HasOption(nuphar::kNupharCodeGenTarget)) {
    if ((target_str == "avx512" && !cpu_id_info.HasAVX512f()) ||
        (target_str == "avx2" && !cpu_id_info.HasAVX2()) ||
        (target_str == "avx" && !cpu_id_info.HasAVX())) {
      LOGS_DEFAULT(WARNING) << "NUPHAR_CODEGEN_TARGET " << target_str
                            << " is not compatible with host machine. "
                               "Target code will be generated, but execution will fail!";
    }
    // For CPU, use target as host since the tvm_host_target_ is the one used to generate code in TVM
    tvm_target_ = tvm::Target::create(codegen_target_->GetTargetName());
    tvm_host_target_ = tvm::Target::create(codegen_target_->GetTargetName());
  } else {
    CreateTVMTarget();
    tvm_host_target_ = tvm::Target::create(GetCurrentHostTargetString());
  }

  tvm_ctx_.device_type = static_cast<DLDeviceType>(tvm_target_->device_type);
  tvm_ctx_.device_id = 0;  // use the default device id for CPU allocator

  whole_graph_shape_infer_ = std::make_shared<ShapeExprContext>();

  AllocatorCreationInfo memory_info(
      [](int /*id*/) {
        return std::make_unique<CPUAllocator>(OrtMemoryInfo("Nuphar", OrtAllocatorType::OrtDeviceAllocator));
      },
      static_cast<OrtDevice::DeviceId>(tvm_ctx_.device_id));

  InsertAllocator(CreateAllocator(memory_info));

  // TODO add multi-target support
  tvm_codegen_manager_ = std::make_unique<TVMCodeGenManager>();

  // Create codegen handle for one target for now
  codegen_handles_.clear();
  auto handle = std::make_unique<NupharCodeGenHandle>();
  tvm_codegen_manager_->Initialization();
  tvm_codegen_manager_->SetCodeGenHandle(handle.get());
  handle->allocator = GetAllocator(tvm_ctx_.device_id, OrtMemTypeDefault);
  handle->codegen_target = codegen_target_.get();
  handle->domain_version_lookup_func =
      [this](const std::string& domain) {
        return GetDomainVersion(domain);
      };

  handle->shape_inference = whole_graph_shape_infer_;

  handle->parallel_min_workloads = std::stoi(settings.GetOptionValue(kNupharParallelMinWorkloads));

  // TODO: remove
  handle->allow_unaligned_buffers = info.allow_unaligned_buffers;  // TODO remove this

  codegen_handles_.push_back(std::move(handle));

  // Runtime Handle
  runtime_handle_ = std::make_unique<nuphar::NupharRuntimeHandle>(tvm_ctx_);
  runtime_handle_->allocator = GetAllocator(tvm_ctx_.device_id, OrtMemTypeDefault);
  runtime_handle_->allow_unaligned_buffers = info.allow_unaligned_buffers;
  runtime_handle_->enable_model_parallelism = false;
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
            [&](const NodeArg& def, size_t) {
              if (def.Shape())
                return Status::OK();
              else
                return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Node: ", node.Name(),
                                       " has no output shape for ", def.Name());
            });
    if (!s.IsOK()) {
      LOGS_DEFAULT(INFO) << "Shape inference incomplete, node execution won't use nuphar provider.";
      LOGS_DEFAULT(INFO) << s.ErrorMessage();
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

  std::set<NodeIndex> nodes_indexes;
  for (auto& node : graph_viewer.Nodes()) {
    nodes_indexes.insert(node.Index());
  }

  std::vector<std::unique_ptr<ComputeCapability>> results;

  typedef std::function<bool(const Node&)> IsSupportedFunc;
  IsSupportedFunc is_supported_func = [&](const Node& node) {
    bool all_shape_defined = true;
    node.ForEachDef([&all_shape_defined](const NodeArg& def, bool /*is_input*/) {
      if (def.Shape() == nullptr) {
        all_shape_defined = false;
      } else {
        for (const auto& dim : def.Shape()->dim()) {
          if (!((utils::HasDimValue(dim) && dim.dim_value() > 0) || utils::HasDimParam(dim)))
            all_shape_defined = false;
        }
      }
    });

    if (!all_shape_defined || !KernelRegistry::HasImplementationOf(*GetKernelRegistryInternal(), node, Type()))
      return false;

    const auto& inputs = node.InputDefs();
    if (node.OpType() == "Tile" && !graph_viewer.IsConstantInitializer(inputs[1]->Name(), true))
      return false;  // do not support tile that has dynamic repeats

    if (node.OpType() == "MaxPool") {
      // TODO: enable support for Indices
      if (node.OutputDefs().size() > 1) {
        return false;
      }
      // TODO: enable support for non-default dilations
      const onnxruntime::NodeAttributes& attrs = node.GetAttributes();
      auto it = attrs.find("dilations");
      if (it != attrs.end()) {
        for (int i = 0; i < it->second.ints_size(); i++) {
          if (it->second.ints(i) > 1) {
            return false;
          }
        }
      }
      // reject when pooling on symbolic dims, since shape computation does not support it yet
      it = attrs.find("kernel_shape");
      ORT_ENFORCE(it != attrs.end());
      int kernel_rank = it->second.ints_size();
      const auto output_shape = node.OutputDefs()[0]->Shape();
      int output_rank = output_shape->dim_size();
      for (int d = output_rank - kernel_rank; d < output_rank; ++d) {
        if (output_shape->dim(d).has_dim_param()) {
          return false;
        }
      }
    }

    if (node.OpType() == "Slice") {
      auto num_inputs = inputs.size();
      ORT_ENFORCE(num_inputs > 0);
      std::vector<int64_t> axes;
      std::vector<int64_t> steps;
      if (num_inputs > 1) {
        // Slice-10
        bool is_starts_dynamic = !graph_viewer.IsConstantInitializer(inputs[1]->Name(), true);
        bool is_ends_dynamic = !graph_viewer.IsConstantInitializer(inputs[2]->Name(), true);
        bool is_axes_dynamic = inputs.size() > 3 && !graph_viewer.IsConstantInitializer(inputs[3]->Name(), true);
        bool is_steps_dynamic = inputs.size() > 4 && !graph_viewer.IsConstantInitializer(inputs[4]->Name(), true);
        if (is_starts_dynamic || is_ends_dynamic || is_axes_dynamic || is_steps_dynamic)
          return false;

        const ONNX_NAMESPACE::TensorProto* steps_tp = nullptr;
        bool found_steps = inputs.size() > 4 && graph_viewer.GetInitializedTensor(inputs[4]->Name(), steps_tp);
        if (found_steps) {
          GetVectorInt64FromTensorProto(steps, *steps_tp);
        }

        const ONNX_NAMESPACE::TensorProto* axes_tp = nullptr;
        bool found_axes = inputs.size() > 3 && graph_viewer.GetInitializedTensor(inputs[3]->Name(), axes_tp);
        if (found_axes) {
          GetVectorInt64FromTensorProto(axes, *axes_tp);
        }
      } else {
        const onnxruntime::NodeAttributes& attrs = node.GetAttributes();
        auto it = attrs.find("axes");
        if (it != attrs.end()) {
          for (int i = 0; i < it->second.ints_size(); i++) {
            axes.push_back(static_cast<int64_t>(it->second.ints(i)));
          }
        }
      }
      // check if we have symbolic dimension on axes
      if (HasUnknownShapeOnAxes(inputs[0], axes))
        return false;
    }

    if (node.OpType() == "Split") {
      const onnxruntime::NodeAttributes& attrs = node.GetAttributes();
      auto axis = std::vector<int64_t>(1);
      auto it = attrs.find("axis");
      if (it != attrs.end()) {
        axis[0] = it->second.i();
      }
      // check if we have symbolic dimension on axis, as TVM split cannot handle that
      if (HasUnknownShapeOnAxes(inputs[0], axis))
        return false;
    }

    if (IsAliasNode(node)) {
      // for AliasNode as final output, skip them to avoid potential copy
      for (auto iter = node.OutputEdgesBegin(); iter != node.OutputEdgesEnd(); ++iter) {
        if (!is_supported_func(iter->GetNode())) {
          return false;
        }
      }
    }

    if (node.OpType() == "Transpose") {
      // When there's symbolic dim in last dim of Transpose output
      // reject it since it was not able to vectorize inlined ops
      const auto output_shape = node.OutputDefs()[0]->Shape();
      const auto output_rank = output_shape->dim_size();
      if (output_rank > 0 && output_shape->dim(output_rank - 1).has_dim_param()) {
        return false;
      }
    }
    return true;
  };
  GraphPartitioner graph_partitioner(is_supported_func);

  ORT_ENFORCE(graph_partitioner.Partition(graph_viewer, per_model_fused_count_, results).IsOK());

  // reset per_model_fused_count_ for main graph, since there might be multiple sessions for subgraphs,
  // this is the time all graph cut should be finished as ORT handles main graph last
  if (!graph_viewer.IsSubgraph())
    per_model_fused_count_ = 0;

  // for any node being fused in results, save initializer tensors
  // because IExecutionProvider::Compile would be called without OpKernelInfo
  const auto& all_initialized_tensors = graph_viewer.GetAllInitializedTensors();

  for (const auto& result : results) {
    for (const auto& node_idx : result->sub_graph->nodes) {
      const auto& node = graph_viewer.GetNode(node_idx);

      node->ForEachDef(
          [this, &all_initialized_tensors, &graph_viewer](const NodeArg& def, bool is_input) {
            if (!is_input)
              return;
            auto iter = all_initialized_tensors.find(def.Name());
            if (iter != all_initialized_tensors.end()) {
              if (graph_viewer.IsConstantInitializer(def.Name(), true)) {
                ORT_ENFORCE(SaveInitializer(def.Name(), iter->second).IsOK());
              }
            }
          });
    }
  }

  if (results.empty()) {
    LOGS_DEFAULT(INFO) << "No node is claimed in nuphar provider.";
  }

  return results;
}

Status NupharExecutionProvider::SaveInitializer(
    const std::string& name,
    const ONNX_NAMESPACE::TensorProto* proto) const {
  auto iter = constant_initializers_used_in_compiled_nodes_.find(name);
  if (iter == constant_initializers_used_in_compiled_nodes_.end()) {
    // create tensor from TensorProto
    // note that session has not call SaveInitializedTensors yet,
    // so we need to make our own copy
    const auto& dims = proto->dims();
    std::vector<int64_t> shape_dims(dims.size());
    for (int i = 0; i < dims.size(); ++i)
      shape_dims[i] = dims[i];

    const TensorShape& shape = TensorShape::ReinterpretBaseType(shape_dims);
    auto data_type = OrtTypeInfo::ElementTypeFromProto(proto->data_type());
    auto t = std::make_unique<Tensor>(
        data_type,
        shape,
        GetAllocator(0, OrtMemTypeDefault)->Alloc(SafeInt<size_t>(shape.Size()) * data_type->Size()),
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

    constant_initializers_used_in_compiled_nodes_.emplace(
        name,
        std::move(t));
  }
  return Status::OK();
}

// Compile nodes into node_compute_funcs
// Here, each of nodes is a fuse node
Status NupharExecutionProvider::Compile(
    const std::vector<onnxruntime::Node*>& nodes,
    std::vector<NodeComputeInfo>& node_compute_funcs) {
  for (const auto* node : nodes) {
    NodeComputeInfo info;

    // Create state function
    // This is similar to the original OpKernel constructor
    // TODO move compilation part out of create_state_func to above
    info.create_state_func =
        [&, node](ComputeContext* ctx, FunctionState* state) {
          std::unique_ptr<NupharKernelState> s =
              std::make_unique<NupharKernelState>(
                  *node,
                  *ctx,
                  *this);

          *state = s.release();
          return 0;
        };

    // Release state function
    // This is similar to the original OpKernel destructor
    info.release_state_func =
        [](FunctionState state) {
          if (state)
            delete static_cast<NupharKernelState*>(state);
        };

    // Compute function
    // This is similar to the original OpKernel's Compute()
    info.compute_func =
        [](FunctionState state, const OrtCustomOpApi*, OrtKernelContext* op_kernel_context) {
          NupharKernelState* s = reinterpret_cast<NupharKernelState*>(state);
          return s->Compute(reinterpret_cast<OpKernelContext*>(op_kernel_context));
        };

    node_compute_funcs.push_back(info);
  }
  // Reset subgraph id counter to avoid same inference session continue the count
  NupharSubgraphUnit::counter = 0;
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
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kNupharExecutionProvider, kOnnxDomain, 1, 10, Gather);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kNupharExecutionProvider, kOnnxDomain, 11, Gather);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kNupharExecutionProvider, kOnnxDomain, 11, GatherElements);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kNupharExecutionProvider, kOnnxDomain, 10, MatMulInteger);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kNupharExecutionProvider, kMSDomain, 1, MatMulInteger16);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kNupharExecutionProvider, kOnnxDomain, 9, 10, Scan);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kNupharExecutionProvider, kOnnxDomain, 11, Scan);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kNupharExecutionProvider, kOnnxDomain, 9, 10, Scatter);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kNupharExecutionProvider, kOnnxDomain, 11, Scatter);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kNupharExecutionProvider, kOnnxDomain, 11, ScatterElements);

static void RegisterStandaloneNupharKernels(KernelRegistry& kernel_registry) {
#define NUPHAR_OP(name, ver, types) \
  ORT_ENFORCE(kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kNupharExecutionProvider, kOnnxDomain, ver, name)>()).IsOK());

#define NUPHAR_VERSIONED_OP(name, start_ver, end_ver, types) \
  ORT_ENFORCE(kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kNupharExecutionProvider, kOnnxDomain, start_ver, end_ver, name)>()).IsOK());

  LIST_NUPHAR_OPS()

#undef NUPHAR_OP
#undef NUPHAR_VERSIONED_OP

  // ops that have multiple type constraints
  ORT_ENFORCE(kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kNupharExecutionProvider, kOnnxDomain, 6, 8, Cast)>()).IsOK());
  ORT_ENFORCE(kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kNupharExecutionProvider, kOnnxDomain, 9, Cast)>()).IsOK());
  ORT_ENFORCE(kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kNupharExecutionProvider, kOnnxDomain, 1, 10, Gather)>()).IsOK());
  ORT_ENFORCE(kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kNupharExecutionProvider, kOnnxDomain, 11, Gather)>()).IsOK());
  ORT_ENFORCE(kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kNupharExecutionProvider, kOnnxDomain, 11, GatherElements)>()).IsOK());
  ORT_ENFORCE(kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kNupharExecutionProvider, kOnnxDomain, 10, MatMulInteger)>()).IsOK());
  ORT_ENFORCE(kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kNupharExecutionProvider, kMSDomain, 1, MatMulInteger16)>()).IsOK());
  ORT_ENFORCE(kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kNupharExecutionProvider, kOnnxDomain, 9, 10, Scan)>()).IsOK());
  ORT_ENFORCE(kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kNupharExecutionProvider, kOnnxDomain, 11, Scan)>()).IsOK());
  ORT_ENFORCE(kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kNupharExecutionProvider, kOnnxDomain, 9, 10, Scatter)>()).IsOK());
  ORT_ENFORCE(kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kNupharExecutionProvider, kOnnxDomain, 11, Scatter)>()).IsOK());
  ORT_ENFORCE(kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kNupharExecutionProvider, kOnnxDomain, 11, ScatterElements)>()).IsOK());
}

std::shared_ptr<KernelRegistry> NupharExecutionProvider::GetKernelRegistryInternal() const {
  if (kernel_registry_ == nullptr) {
    kernel_registry_ = std::make_shared<KernelRegistry>();
    RegisterStandaloneNupharKernels(*kernel_registry_);
  }
  return kernel_registry_;
}

}  // namespace onnxruntime
