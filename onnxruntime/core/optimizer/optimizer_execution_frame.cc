
#include "core/common/common.h"
#include "core/common/status.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/macros.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/data_types.h"
#include "core/framework/mldata_type_utils.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/fuse_nodes_funcs.h"
#include "core/optimizer/optimizer_execution_frame.h"

namespace onnxruntime {

OptimizerExecutionFrame::Info::Info(const std::vector<const Node*>& nodes,
                                    const InitializedTensorSet& initialized_tensor_set) {
  // Create a CPU execution provider
  CPUExecutionProviderInfo info;
  cpu_execution_provider_ = std::make_unique<CPUExecutionProvider>(info);
  allocator_ptr_ = cpu_execution_provider_->GetAllocator(device_id_, mem_type_);
  ORT_ENFORCE(allocator_ptr_ != nullptr, "Failed to get allocator for optimizer");

  // Create MLValues related maps
  auto initialize_maps = [this, initialized_tensor_set](const NodeArg& arg, size_t index) -> Status {
    int idx = mlvalue_name_idx_map_.Add(arg.Name());
    mlvalue_idx_nodearg_map_[idx] = &arg;

    // Only create MLValue instances for initializers used by an array of nodes.
    InitializedTensorSet::const_iterator it = initialized_tensor_set.find(arg.Name());
    if (it != initialized_tensor_set.cend()) {
      MLValue mlvalue;
      utils::TensorProtoToMLValue(*(it->second), allocator_ptr_, nullptr, 0, mlvalue);
      initializers_[idx] = mlvalue;
    }

    return Status::OK();
  };

  for (auto* node : nodes) {
    onnxruntime::Node::ForEachWithIndex(node->InputDefs(), initialize_maps);
    onnxruntime::Node::ForEachWithIndex(node->OutputDefs(), initialize_maps);
  }

  node_index_info_ = std::make_unique<NodeIndexInfo>(nodes, mlvalue_name_idx_map_);

  // create kernels for these nodes
  for (auto* node : nodes) {
    std::unique_ptr<OpKernel> op_kernel;
    std::shared_ptr<KernelRegistry> kernel_registry = cpu_execution_provider_->GetKernelRegistry();
    auto status = kernel_registry->TryCreateKernel(*node,
                                                   *cpu_execution_provider_,
                                                   initializers_,
                                                   mlvalue_name_idx_map_,
                                                   FuncManager(),
                                                   op_kernel);
    kernels_[node->Index()] = std::move(op_kernel);
  }
}

const OpKernel* OptimizerExecutionFrame::Info::GetKernel(NodeIndex node_id) const {
  if (kernels_.count(node_id) == 0) {
    return nullptr;
  }

  return kernels_.find(node_id)->second.get();
}

// For optimizer, probably no need to pass feed_mlvalue_idxs, feeds to initialize IExecutionFrame.
// If needed, the parameters of OptimizerExecutionFrame ctor can be changed later.
OptimizerExecutionFrame::OptimizerExecutionFrame(const Info& info,
                                                 const std::vector<int>& fetch_mlvalue_idxs)
    : IExecutionFrame(std::vector<int>(),
                      std::vector<MLValue>(),
                      info.GetInitializers(),
                      fetch_mlvalue_idxs,
                      std::vector<MLValue>(),
                      info.GetMLValueNameIdxMap(),
                      info.GetNodeIndexInfo()),
                      info_(info) {
}

OptimizerExecutionFrame::~OptimizerExecutionFrame() = default;

AllocatorPtr OptimizerExecutionFrame::GetAllocatorImpl(const OrtAllocatorInfo& info) const {
  return info_.GetAllocator(info);
}

// This method is not thread safe!
// Return S_OK and nullptr if index map to an value that is an unused optional input/output
Status OptimizerExecutionFrame::CreateNodeOutputMLValueImpl(MLValue& mlvalue, int mlvalue_idx, const TensorShape* shape) {
  const DataTypeImpl* ml_type = utils::GetMLDataType(*(info_.GetMLValueIdxNodeArgMap().at(mlvalue_idx)));
  if (ml_type == nullptr)
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "Tried to allocate without valid type information, mlvalue index=" + std::to_string(mlvalue_idx));
  if (!ml_type->IsTensorType()) {
    const NonTensorTypeBase* non_tensor_type = static_cast<const NonTensorTypeBase*>(ml_type);
    auto creator = non_tensor_type->GetCreateFunc();
    mlvalue.Init(creator(),
                 non_tensor_type,
                 non_tensor_type->GetDeleteFunc());
    return Status::OK();
  }

  // tensors
  auto element_type = static_cast<const TensorTypeBase*>(ml_type)->GetElementType();
  AllocatorPtr allocator_ptr = info_.GetAllocator();
  OrtAllocatorInfo allocator_info = allocator_ptr->Info();

  int64_t len = shape->Size();
  if (len < 0) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Tensor shape cannot contain any negative value");
  }
  size_t size;
  if (!IAllocator::CalcMemSizeForArrayWithAlignment<64>(len, element_type->Size(), &size)) {
    return Status(common::ONNXRUNTIME, common::FAIL, "size overflow");
  }
  
  void* buffer = size == 0 ? nullptr : allocator_ptr->Alloc(size);
  std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(element_type,
                                                              *shape,
                                                              buffer,
                                                              allocator_info,
                                                              allocator_ptr);

  mlvalue.Init(p_tensor.release(),
               DataTypeImpl::GetType<Tensor>(),
               DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());

  return Status::OK();
}

}