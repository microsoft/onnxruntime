
#include "core/common/common.h"
#include "core/common/status.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/macros.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/data_types.h"
#include "core/framework/mldata_type_utils.h"
#include "core/graph/graph_viewer.h"
#include "core/optimizer/graph_optimizer.h"

namespace onnxruntime {

// Build the MLValue name->idx mapping
common::Status GraphOptimizer::InitMLValueNameIndexMapping() {
  LOGS(logger_, INFO) << "SaveMLValueNameIndexMapping";
  int idx = 0;

  // we keep all graph inputs (including initializers), even if they are unused, so make sure they all have an entry
  for (const auto* input_def : graph_.GetInputsIncludingInitializers()) {
    idx = mlvalue_name_idx_map_.Add(input_def->Name());
    VLOGS(logger_, 1)
        << "Added graph input with name: " << input_def->Name() << " to MLValueIndex with index: " << idx;
  }

  for (auto& node : graph_.Nodes()) {
    // build the MLValue->index map
    for (const auto* input_def : node.InputDefs()) {
      if (input_def->Exists()) {
        idx = mlvalue_name_idx_map_.Add(input_def->Name());
        VLOGS(logger_, 1)
            << "Added input argument with name: " << input_def->Name() << " to MLValueIndex with index: " << idx;
      }
    }

    for (const auto* input_def : node.ImplicitInputDefs()) {
      if (input_def->Exists()) {
        idx = mlvalue_name_idx_map_.Add(input_def->Name());
        VLOGS(logger_, 1)
            << "Added implicit input argument with name: " << input_def->Name() << " to MLValueIndex with index: " << idx;
      }
    }

    for (const auto* output_def : node.OutputDefs()) {
      if (output_def->Exists()) {
        mlvalue_name_idx_map_.Add(output_def->Name());
        VLOGS(logger_, 1)
            << "Added output argument with name: " << output_def->Name() << " to MLValueIndex with index: " << idx;
      }
    }
  }

  // allocate MLValue for graph outputs when coming from initializers
  for (const auto& output : graph_.GetOutputs()) {
    if (output->Exists()) {
      idx = mlvalue_name_idx_map_.Add(output->Name());
      VLOGS(logger_, 1)
          << "Added graph output with name: " << output->Name() << " to MLValueIndex with index: " << idx;
    }
  }

  LOGS(logger_, INFO) << "Done saving MLValue mappings.";
  return Status::OK();
}

Status GraphOptimizer::InitMLValues() {
  // Resize the all_value_ vector
  all_values_.resize(mlvalue_name_idx_map_.MaxIdx() + 1);

  // De-serialize initializers
  const onnxruntime::InitializedTensorSet& initialized_tensor_set = graph_.GetAllInitializedTensors();
  for (const auto& entry : initialized_tensor_set) {
    const std::string& name = entry.first;
    int mlvalue_index;
    ORT_RETURN_IF_ERROR(mlvalue_name_idx_map_.GetIdx(name, mlvalue_index));
    VLOGS(logger_, 1) << "About to add weight with name: " << name << " and index: " << mlvalue_index;

    MLValue mlvalue;
    utils::TensorProtoToMLValue(*(entry.second), allocator_ptr_, nullptr, 0, mlvalue);

    // save mlvalue in a map
    all_values_[mlvalue_index] = mlvalue;
    VLOGS(logger_, 1) << "Added weight with name : " << name << " with index: " << mlvalue_index;
  }
  return Status::OK();
}

Status GraphOptimizer::Init() {
  GraphViewer viewer(graph_);

  // Create a CPU execution provider
  LOGS(logger_, INFO) << "Adding default CPU execution provider.";
  CPUExecutionProviderInfo info;
  cpu_execution_provider_ = std::make_unique<CPUExecutionProvider>(info);
  allocator_ptr_ = cpu_execution_provider_->GetAllocator(device_id_, mem_type_);
  if (!allocator_ptr_) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Failed to get allocator for optimizer");
  }

  // Init MLValueNameIdxMap
  InitMLValueNameIndexMapping();

  // Init MLValues;
  InitMLValues();

  return Status::OK();
}

const MLValue* GraphOptimizer::GetNodeInputOrOutputMLValue(const std::string& mlvalue_name) const {
  int mlvalue_idx = -1;
  mlvalue_name_idx_map_.GetIdx(mlvalue_name, mlvalue_idx);
  return mlvalue_idx != -1 ? &all_values_[mlvalue_idx] : nullptr;
}

MLValue* GraphOptimizer::GetMutableNodeInputOrOutputMLValue(const std::string& mlvalue_name) {
  return const_cast<MLValue*>(GetNodeInputOrOutputMLValue(mlvalue_name));
}

Status GraphOptimizer::GetOrCreateNodeOutputMLValue(const NodeArg* node_arg,
                                                    const MLValueAllocationParameters& parameters,
                                                    MLValue*& p_mlvalue) {
  int mlvalue_idx = -1;
  mlvalue_name_idx_map_.GetIdx(node_arg->Name(), mlvalue_idx);

  // return nullptr if it is optional
  if (mlvalue_idx == -1) {
    p_mlvalue = nullptr;
    return Status::OK();
  }

  p_mlvalue = &all_values_.at(mlvalue_idx);
  if (p_mlvalue->IsAllocated()) {
    VerifyShape(p_mlvalue, parameters);
    return Status::OK();
  }

  const DataTypeImpl* ml_type = utils::GetMLDataType(*node_arg);
  if (ml_type == nullptr)
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "Tried to allocate without valid type information, mlvalue index=" + std::to_string(mlvalue_idx));
  if (!ml_type->IsTensorType()) {
    const NonTensorTypeBase* non_tensor_type = static_cast<const NonTensorTypeBase*>(ml_type);
    auto creator = non_tensor_type->GetCreateFunc();
    p_mlvalue->Init(creator(),
                    non_tensor_type,
                    non_tensor_type->GetDeleteFunc());
    return Status::OK();
  }

  // tensors
  auto element_type = static_cast<const TensorTypeBase*>(ml_type)->GetElementType();
  const TensorShape& shape = parameters.GetTensorShape();
  OrtAllocatorInfo allocator_info = allocator_ptr_->Info();

  size_t size;
  {
    int64_t len = shape.Size();
    if (len < 0) {
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Tensor shape cannot contain any negative value");
    }
    if (!IAllocator::CalcMemSizeForArrayWithAlignment<64>(len, element_type->Size(), &size)) {
      return Status(common::ONNXRUNTIME, common::FAIL, "size overflow");
    }
  }

  void* buffer = size == 0 ? nullptr : allocator_ptr_->Alloc(size);
  std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(element_type,
                                                              shape,
                                                              buffer,
                                                              allocator_info,
                                                              allocator_ptr_);

  p_mlvalue->Init(p_tensor.release(),
                  DataTypeImpl::GetType<Tensor>(),
                  DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());

  return Status::OK();
}

}