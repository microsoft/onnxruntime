// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/optimizer_execution_frame.h"

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/macros.h"
#include "core/common/status.h"
#include "core/framework/callback.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/data_types.h"
#include "core/framework/fuse_nodes_funcs.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/kernel_type_str_resolver.h"
#include "core/framework/mldata_type_utils.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/TensorSeq.h"

namespace onnxruntime {

static size_t EstimateInputsOutputs(gsl::span<const Node* const> nodes) {
  size_t num = 0;
  for (auto n : nodes) {
    num += n->InputDefs().size() + n->OutputDefs().size();
  }
  return num;
}

OptimizerExecutionFrame::Info::Info(const std::vector<const Node*>& nodes,
                                    const InitializedTensorSet& initialized_tensor_set,
                                    const Path& model_path,
                                    const IExecutionProvider& execution_provider,
                                    const std::function<bool(const std::string&)>& is_sparse_initializer_func)
    : execution_provider_(execution_provider),
      is_sparse_initializer_func_(is_sparse_initializer_func) {
  allocator_ptr_ = std::make_shared<CPUAllocator>();
  ORT_ENFORCE(allocator_ptr_, "Failed to get allocator for optimizer");

  ORT_THROW_IF_ERROR(data_transfer_mgr_.RegisterDataTransfer(std::make_unique<CPUDataTransfer>()));

  // Create MLValues related maps
  auto initialize_maps = [this, &initialized_tensor_set, &model_path](const NodeArg& arg, size_t /*index*/) -> Status {
    int idx = ort_value_name_idx_map_.Add(arg.Name());
    ort_value_idx_nodearg_map_.insert_or_assign(idx, &arg);

    // Only create OrtValue instances for initializers used by an array of nodes.
    InitializedTensorSet::const_iterator it = initialized_tensor_set.find(arg.Name());
    if (it != initialized_tensor_set.cend()) {
      const auto& tensor_proto = *(it->second);
      OrtValue ort_value;
      ORT_RETURN_IF_ERROR(
          utils::TensorProtoToOrtValue(Env::Default(),
                                       model_path.IsEmpty() ? nullptr : model_path.ToPathString().c_str(),
                                       tensor_proto, allocator_ptr_, ort_value));

      initializers_[idx] = std::move(ort_value);
    }

    return Status::OK();
  };

  // TODO: node->ImplicitInputDefs() need to be added here for control flow nodes.
  auto num_inputs_outputs = EstimateInputsOutputs(nodes);
  ort_value_name_idx_map_.Reserve(num_inputs_outputs);
  ort_value_idx_nodearg_map_.reserve(num_inputs_outputs);
  initializers_.reserve(initialized_tensor_set.size());

  for (auto* node : nodes) {
    ORT_THROW_IF_ERROR(onnxruntime::Node::ForEachWithIndex(node->InputDefs(), initialize_maps));
    ORT_THROW_IF_ERROR(onnxruntime::Node::ForEachWithIndex(node->OutputDefs(), initialize_maps));
  }

  node_index_info_ = std::make_unique<NodeIndexInfo>(nodes, ort_value_name_idx_map_);
}

OptimizerExecutionFrame::Info::Info(const std::vector<const Node*>& nodes,
                                    const std::unordered_map<std::string, OrtValue>& initialized_tensor_set,
                                    const Path& model_path,
                                    const IExecutionProvider& execution_provider,
                                    const std::function<bool(const std::string&)>& is_sparse_initializer_func)
    : execution_provider_(execution_provider),
      is_sparse_initializer_func_(is_sparse_initializer_func) {
  allocator_ptr_ = std::make_shared<CPUAllocator>();
  ORT_ENFORCE(allocator_ptr_, "Failed to get allocator for optimizer");

  ORT_THROW_IF_ERROR(data_transfer_mgr_.RegisterDataTransfer(std::make_unique<CPUDataTransfer>()));

  // Create MLValues related maps
  auto initialize_maps = [this, &initialized_tensor_set, &model_path](const NodeArg& arg, size_t /*index*/) -> Status {
    (void)model_path;
    int idx = ort_value_name_idx_map_.Add(arg.Name());
    ort_value_idx_nodearg_map_.insert_or_assign(idx, &arg);

    // Only create OrtValue instances for initializers used by an array of nodes.
    auto it = initialized_tensor_set.find(arg.Name());
    if (it != initialized_tensor_set.cend()) {
      initializers_[idx] = it->second;
    }
    return Status::OK();
  };

  // TODO: node->ImplicitInputDefs() need to be added here for control flow nodes.
  auto num_inputs_outputs = EstimateInputsOutputs(nodes);
  ort_value_name_idx_map_.Reserve(num_inputs_outputs);
  ort_value_idx_nodearg_map_.reserve(num_inputs_outputs);
  initializers_.reserve(initialized_tensor_set.size());

  for (auto* node : nodes) {
    ORT_THROW_IF_ERROR(onnxruntime::Node::ForEachWithIndex(node->InputDefs(), initialize_maps));
    ORT_THROW_IF_ERROR(onnxruntime::Node::ForEachWithIndex(node->OutputDefs(), initialize_maps));
  }

  node_index_info_ = std::make_unique<NodeIndexInfo>(nodes, ort_value_name_idx_map_);
}

Status OptimizerExecutionFrame::Info::TryFindKernel(const Node* node, const KernelCreateInfo** out) const {
  std::shared_ptr<KernelRegistry> kernel_registry = execution_provider_.GetKernelRegistry();
  const OpSchemaKernelTypeStrResolver kernel_type_str_resolver{};
  return kernel_registry->TryFindKernel(*node, execution_provider_.Type(), kernel_type_str_resolver, out);
}

static Status TryCreateKernel(const Node& node,
                              const KernelRegistry& kernel_registry,
                              const IExecutionProvider& execution_provider,
                              const std::unordered_map<int, OrtValue>& constant_initialized_tensors,
                              const OrtValueNameIdxMap& ort_value_name_idx_map,
                              FuncManager& funcs_mgr,
                              const DataTransferManager& data_transfer_mgr,
                              const ConfigOptions& config_options,
                              /*out*/ std::unique_ptr<OpKernel>& op_kernel) {
  const OpSchemaKernelTypeStrResolver kernel_type_str_resolver{};
  const KernelCreateInfo* kernel_create_info = nullptr;
  ORT_RETURN_IF_ERROR(kernel_registry.TryFindKernel(node, execution_provider.Type(), kernel_type_str_resolver,
                                                    &kernel_create_info));

  static const AllocatorMap dummy_allocators;

  OpKernelInfo kernel_info(node,
                           *kernel_create_info->kernel_def,
                           execution_provider,
                           constant_initialized_tensors,
                           ort_value_name_idx_map,
                           data_transfer_mgr,
                           dummy_allocators,
                           config_options);

  return kernel_create_info->kernel_create_func(funcs_mgr, kernel_info, op_kernel);
}

std::unique_ptr<const OpKernel>
OptimizerExecutionFrame::Info::CreateKernel(const Node* node, const ConfigOptions& config_options) const {
  std::unique_ptr<OpKernel> op_kernel;
  std::shared_ptr<KernelRegistry> kernel_registry = execution_provider_.GetKernelRegistry();
  FuncManager func;
  auto status = TryCreateKernel(*node, *kernel_registry, execution_provider_, initializers_,
                                ort_value_name_idx_map_, func, data_transfer_mgr_, config_options,
                                op_kernel);

  // Kernel found in the CPU kernel registry
  if (status.IsOK())
    return std::unique_ptr<const OpKernel>(std::move(op_kernel));

  // No kernel found in the CPU kernel registry
  return nullptr;
}

// For optimizer, probably no need to pass feed_mlvalue_idxs, feeds to initialize IExecutionFrame.
// If needed, the parameters of OptimizerExecutionFrame ctor can be changed later.
OptimizerExecutionFrame::OptimizerExecutionFrame(const Info& info,
                                                 const std::vector<int>& fetch_mlvalue_idxs,
                                                 const std::vector<OrtValue>& fetches)
    : IExecutionFrame(info.GetMLValueNameIdxMap(), info.GetNodeIndexInfo(), fetch_mlvalue_idxs),
      info_(info) {
  Init(gsl::span<const int>(), gsl::span<const OrtValue>(), info.GetInitializers(), info.GetSparseInitializerLookupFunc(), fetches);
}

AllocatorPtr OptimizerExecutionFrame::GetAllocatorImpl(const OrtDevice&) const {
  return info_.GetAllocator();
}

Status OptimizerExecutionFrame::CopyTensor(const Tensor& src, Tensor& dest) const {
  return info_.GetDataTransferManager().CopyTensor(src, dest);
}

const DataTransferManager& OptimizerExecutionFrame::GetDataTransferManager() const {
  return info_.GetDataTransferManager();
}

// This method is not thread safe!
// Return S_OK and nullptr if index map to an value that is an unused optional input/output
Status OptimizerExecutionFrame::CreateNodeOutputMLValueImpl(OrtValue& ort_value, int ort_value_idx, const TensorShape* shape) {
  const DataTypeImpl* ml_type = utils::GetMLDataType(*(info_.GetMLValueIdxNodeArgMap().at(ort_value_idx)));
  if (ml_type == nullptr)
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "Tried to allocate without valid type information, ort_value index=" + std::to_string(ort_value_idx));
  if (ml_type->IsSparseTensorType()) {
#if !defined(DISABLE_SPARSE_TENSORS)
    auto element_type = ml_type->AsSparseTensorType()->GetElementType();
    SparseTensor::InitOrtValue(element_type, *shape, info_.GetAllocator(), ort_value);
    return Status::OK();
#else
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Sparse tensor is not supported in this build");
#endif
  }

  if (ml_type->IsTensorSequenceType()) {
    auto element_type = ml_type->AsSequenceTensorType()->GetElementType();
    auto p_sequence = std::make_unique<TensorSeq>(element_type);
    auto ml_tensor_sequence = DataTypeImpl::GetType<TensorSeq>();
    ort_value.Init(p_sequence.release(), ml_tensor_sequence, ml_tensor_sequence->GetDeleteFunc());
    return Status::OK();
  }

  if (!ml_type->IsTensorType()) {
    assert(ml_type->AsNonTensorType() != nullptr);
    const NonTensorTypeBase* non_tensor_type = static_cast<const NonTensorTypeBase*>(ml_type);
    auto creator = non_tensor_type->GetCreateFunc();
    ort_value.Init(creator(), non_tensor_type, non_tensor_type->GetDeleteFunc());
    return Status::OK();
  }

  // tensors
  auto element_type = static_cast<const TensorTypeBase*>(ml_type)->GetElementType();
  AllocatorPtr allocator_ptr = info_.GetAllocator();
  Tensor::InitOrtValue(element_type, *shape, std::move(allocator_ptr), ort_value);
  return Status::OK();
}

}  // namespace onnxruntime
