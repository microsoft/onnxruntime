// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/execution_frame.h"

#include <sstream>

#include "core/framework/mem_pattern_planner.h"
#include "core/framework/ml_value_patterns_planner.h"
#include "core/framework/node_index_info.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/framework/utils.h"

using namespace onnxruntime::common;

namespace onnxruntime {

IExecutionFrame::IExecutionFrame(const std::vector<int>& feed_mlvalue_idxs,
                                 const std::vector<MLValue>& feeds,
                                 const std::unordered_map<int, MLValue>& initializers,
                                 const std::vector<int>& fetch_mlvalue_idxs,
                                 const std::vector<MLValue>& fetches,
                                 const MLValueNameIdxMap& mlvalue_idx_map,
                                 const NodeIndexInfo& node_index_info)
    : node_index_info_{node_index_info}, fetch_mlvalue_idxs_{fetch_mlvalue_idxs} {
  ORT_ENFORCE(feeds.size() == feed_mlvalue_idxs.size());
  ORT_ENFORCE(fetches.empty() || fetches.size() == fetch_mlvalue_idxs.size());

  Init(feed_mlvalue_idxs, feeds, initializers, fetch_mlvalue_idxs, fetches, mlvalue_idx_map);
}

IExecutionFrame::~IExecutionFrame() = default;

// Return nullptr if index map to an value that is an unused optional input/output
const MLValue* IExecutionFrame::GetNodeInputOrOutputMLValue(int index) const {
  int mlvalue_idx = GetNodeIdxToMLValueIdx(index);
  return mlvalue_idx != NodeIndexInfo::kInvalidEntry ? &all_values_[mlvalue_idx] : nullptr;
}

MLValue* IExecutionFrame::GetMutableNodeInputOrOutputMLValue(int index) {
  return const_cast<MLValue*>(GetNodeInputOrOutputMLValue(index));
}

// TO DO: make it thread safe
// This method is not thread safe!
// Return S_OK and nullptr if index map to an value that is an unused optional input/output
Status IExecutionFrame::GetOrCreateNodeOutputMLValue(int index, const TensorShape* shape, MLValue*& p_mlvalue) {
  auto status = Status::OK();
  int mlvalue_idx = GetNodeIdxToMLValueIdx(index);

  // return nullptr if it is optional
  if (mlvalue_idx == NodeIndexInfo::kInvalidEntry) {
    p_mlvalue = nullptr;
  } else {
    p_mlvalue = &all_values_[mlvalue_idx];

    if (p_mlvalue->IsAllocated()) {
      // already allocated. verify shape matches if tensor.
      if (p_mlvalue->IsTensor()) {
        const Tensor& tensor = p_mlvalue->Get<Tensor>();
        ORT_ENFORCE(shape && tensor.Shape() == *shape,
                    "MLValue shape verification failed. Current shape:", tensor.Shape(),
                    " Requested shape:", shape ? shape->ToString() : "null");
      }
    } else {
      status = CreateNodeOutputMLValueImpl(*p_mlvalue, mlvalue_idx, shape);
    }
  }

  return status;
}

AllocatorPtr IExecutionFrame::GetAllocator(const OrtAllocatorInfo& info) const {
  return GetAllocatorImpl(info);
}

Status IExecutionFrame::ReleaseMLValue(int mlvalue_idx) {
  return ReleaseMLValueImpl(mlvalue_idx);
}

Status IExecutionFrame::ReleaseMLValueImpl(int mlvalue_idx) {
  if (mlvalue_idx == NodeIndexInfo::kInvalidEntry || static_cast<size_t>(mlvalue_idx) >= all_values_.size()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "invalid index ", mlvalue_idx);
  }

  all_values_[mlvalue_idx] = MLValue();
  return Status::OK();
}

int IExecutionFrame::GetNodeIdxToMLValueIdx(int index) const {
  int mlvalue_idx = node_index_info_.GetMLValueIndex(index);
  ORT_ENFORCE(mlvalue_idx == NodeIndexInfo::kInvalidEntry ||
              (mlvalue_idx >= 0 && static_cast<size_t>(mlvalue_idx) < all_values_.size()));

  return mlvalue_idx;
}

void IExecutionFrame::Init(const std::vector<int>& feed_mlvalue_idxs,
                           const std::vector<MLValue>& feeds,
                           const std::unordered_map<int, MLValue>& initializers,
                           const std::vector<int>& fetch_mlvalue_idxs,
                           const std::vector<MLValue>& fetches,
                           const MLValueNameIdxMap& mlvalue_idx_map) {
  // 1. resize the all_value_ vector
  all_values_.resize(mlvalue_idx_map.MaxIdx() + 1);

  // 2. Handle non-empty output vector
  if (!fetches.empty()) {
    auto num_fetches = fetch_mlvalue_idxs.size();

    for (size_t idx = 0; idx < num_fetches; ++idx) {
      int mlvalue_idx = fetch_mlvalue_idxs[idx];
      all_values_[mlvalue_idx] = fetches[idx];
    }
  }

  // 3. handle the weights.
  // We do this after the fetches to handle an edge case (possibly dubious) where a Constant is an output.
  // The Constant gets lifted to an initializer so there's no Node producing the value as an output during Graph
  // execution (i.e. Graph execution won't write the value to all_values_).
  // A non-empty fetches vector will overwrite the actual weight in all_values_[mlvalue_idx] if we did this earlier.
  // This makes the ONNX Constant test (onnx\backend\test\data\node\test_constant) happy as that
  // involves a graph with a single Constant node.
  for (const auto& entry : initializers) {
    int mlvalue_index = entry.first;
    all_values_[mlvalue_index] = entry.second;
  }

  // 4. handle feed in values. these can override initializer values so must be last
  for (size_t idx = 0, end = feed_mlvalue_idxs.size(); idx < end; ++idx) {
    int mlvalue_idx = feed_mlvalue_idxs[idx];
    // we are sharing the underline tensor/object for MLValue
    all_values_[mlvalue_idx] = feeds[idx];
  }
}

Status IExecutionFrame::GetOutputs(std::vector<MLValue>& fetches) {
  auto num_fetches = fetch_mlvalue_idxs_.size();

  if (fetches.empty()) {
    fetches.resize(num_fetches);
  } else {
    // if there's a mismatch things are out so sync so fail
    if (fetches.size() != num_fetches) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Fetches vector passed to GetOutputs contains ", fetches.size(),
                             " entries which doesn't match the number of fetches the frame was initialized with of ",
                             num_fetches);
    }
  }

  for (size_t idx = 0; idx < num_fetches; ++idx) {
    fetches[idx] = GetMLValue(fetch_mlvalue_idxs_[idx]);
  }

  return Status::OK();
}

bool IExecutionFrame::IsOutput(int mlvalue_idx) const {
  return std::find(fetch_mlvalue_idxs_.begin(), fetch_mlvalue_idxs_.end(), mlvalue_idx) != fetch_mlvalue_idxs_.end();
}

ExecutionFrame::ExecutionFrame(const std::vector<int>& feed_mlvalue_idxs,
                               const std::vector<MLValue>& feeds,
                               const std::vector<int>& fetch_mlvalue_idxs,
                               const std::vector<MLValue>& fetches,
                               const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                               const SessionState& session_state)
    : IExecutionFrame(feed_mlvalue_idxs, feeds, session_state.GetInitializedTensors(), fetch_mlvalue_idxs, fetches,
                      session_state.GetMLValueNameIdxMap(), session_state.GetNodeIndexInfo()),
      session_state_{session_state},
      mem_patterns_{nullptr},
      planner_{nullptr} {
  // map the custom allocators to mlvalue_idx entries
  if (!fetch_allocators.empty()) {
    for (size_t idx = 0, end = fetch_mlvalue_idxs.size(); idx < end; ++idx) {
      int mlvalue_idx = fetch_mlvalue_idxs[idx];

      auto custom_alloc_entry = fetch_allocators.find(idx);
      if (custom_alloc_entry != fetch_allocators.cend()) {
        custom_allocators_[mlvalue_idx] = custom_alloc_entry->second;
      }
    }
  }

  // If the session enable memory pattern optimization
  // and we have execution plan generated, try to setup
  // memory pattern optimization.
  if (session_state.GetExecutionPlan()) {
    std::vector<TensorShape> input_shapes;
    bool all_tensors = true;
    for (const auto& feed : feeds) {
      if (!(feed.IsTensor())) {
        all_tensors = false;
        break;
      }
      auto& tensor = feed.Get<Tensor>();
      input_shapes.push_back(tensor.Shape());
    }

    // if there are some traditional ml value type in inputs disable the memory pattern optimization.
    if (all_tensors) {
      mem_patterns_ = session_state.GetMemoryPatternGroup(input_shapes);
      // if no existing patterns, generate one in this executionframe
      if (!mem_patterns_) {
        planner_ = std::make_unique<MLValuePatternPlanner>(*session_state.GetExecutionPlan());
      } else {
        // pre-allocate the big chunk requested in memory pattern.
        // all the internal kernel's input/output tensors will be allocated on these buffer.
        for (size_t i = 0; i < mem_patterns_->locations.size(); i++) {
          ORT_ENFORCE(buffers_.find(mem_patterns_->locations[i]) == buffers_.end());
          AllocatorPtr alloc = GetAllocator(mem_patterns_->locations[i]);
          void* buffer = mem_patterns_->patterns[i].PeakSize() > 0
                             ? alloc->Alloc(mem_patterns_->patterns[i].PeakSize())
                             : nullptr;
          buffers_[mem_patterns_->locations[i]] = BufferUniquePtr(buffer, alloc);
        }
      }
    }
  }
}

ExecutionFrame::~ExecutionFrame() = default;

Status ExecutionFrame::AllocateMLValueTensorSelfOwnBuffer(MLValue& mlvalue,
                                                          int mlvalue_index,
                                                          MLDataType element_type,
                                                          const OrtAllocatorInfo& location,
                                                          const TensorShape& shape,
                                                          bool create_fence) {
  return AllocateMLValueTensorSelfOwnBufferHelper(mlvalue, mlvalue_index, element_type, location, shape, create_fence);
}

Status ExecutionFrame::AllocateMLValueTensorSelfOwnBufferHelper(MLValue& mlvalue,
                                                                int mlvalue_index,
                                                                MLDataType element_type,
                                                                const OrtAllocatorInfo& location,
                                                                const TensorShape& shape,
                                                                bool create_fence) {
  if (mlvalue_index == NodeIndexInfo::kInvalidEntry) {
    return Status(ONNXRUNTIME, FAIL, "Trying to allocate memory for unused optional inputs/outputs");
  }

  size_t size;
  int64_t len = shape.Size();
  if (len < 0) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Tensor shape cannot contain any negative value");
  }
  if (!IAllocator::CalcMemSizeForArrayWithAlignment<64>(len, element_type->Size(), &size)) {
    return Status(ONNXRUNTIME, FAIL, "size overflow");
  }

  auto alloc = GetAllocator(location);

  // create fence if needed
  if (create_fence) {
    ORT_ENFORCE(mlvalue.Fence() == nullptr);
    FencePtr f = alloc->CreateFence(&session_state_);
    // it is OK to have fence been nullptr if the execution provider has no async execution,
    // and allocator::CreateFence returns nullptr
    mlvalue.SetFence(f);
  }

  // if we have pre-calculated memory pattern, and the mlvalue is not output mlvalue
  // try to allocated on pre-allocated big chunk.
  const auto& per_alloc_plan = GetAllocationPlan(mlvalue_index);
  if (mem_patterns_ && per_alloc_plan.alloc_kind != AllocKind::kAllocateOutput) {
    auto pattern = mem_patterns_->GetPatterns(location);
    if (pattern) {
      auto block = pattern->GetBlock(mlvalue_index);
      // if block not found, fall back to default behavior
      if (block) {
        auto it = buffers_.find(location);
        // if the block is not correct, log message then fall back to default behavior
        if (it != buffers_.end() && block->size_ == size) {
          void* buffer = it->second.get();
          auto status = AllocateTensorWithPreAllocateBufferHelper(
              mlvalue, static_cast<void*>(static_cast<char*>(buffer) + block->offset_),
              element_type, location, shape);
          return status;
        }
        if (block->size_ != size) {
          LOGS_DEFAULT(WARNING) << "For mlvalue with index: " << mlvalue_index << ", block in memory pattern size is: "
                                << block->size_ << " but the actually size is: " << size
                                << ", fall back to default allocation behavior";
        } else if (it == buffers_.end()) {
          LOGS_DEFAULT(WARNING) << "For mlvalue with index: " << mlvalue_index
                                << ", block not found in target location. fall back to default allocation behavior";
        }
      }
    }
  }
  //no memory pattern, or the pattern is not correct.
  std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(element_type, shape, alloc);

  mlvalue.Init(p_tensor.release(),
               DataTypeImpl::GetType<Tensor>(),
               DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());

  // trace the memory allocation.
  // don't trace the memory allocation on string tensors, as it need
  // placement new, we don't support it in memory pattern optimization.
  if (element_type != DataTypeImpl::GetType<std::string>()) {
    TraceAllocate(mlvalue_index, size);
  }

  return Status::OK();
}

Status ExecutionFrame::AllocateMLValueTensorPreAllocateBuffer(MLValue& mlvalue,
                                                              int mlvalue_index_reuse,
                                                              MLDataType element_type,
                                                              const OrtAllocatorInfo& location,
                                                              const TensorShape& shape,
                                                              bool create_fence) {
  MLValue& mlvalue_reuse = GetMutableMLValue(mlvalue_index_reuse);

  auto* reuse_tensor = mlvalue_reuse.GetMutable<Tensor>();
  void* reuse_buffer = reuse_tensor->MutableDataRaw();

  // create fence on reused mlvalue if needed
  // TODO: differentiate reuse and alias, by add AllocKind::kAlias?
  if (create_fence && mlvalue_reuse.Fence() == nullptr) {
    FencePtr f = GetAllocator(location)->CreateFence(&session_state_);
    mlvalue_reuse.SetFence(f);
  }

  // reused MLValue share the same fence
  mlvalue.ShareFenceWith(mlvalue_reuse);
  return AllocateTensorWithPreAllocateBufferHelper(mlvalue, reuse_buffer, element_type, location, shape);
}

Status ExecutionFrame::AllocateTensorWithPreAllocateBufferHelper(MLValue& mlvalue,
                                                                 void* pBuffer,
                                                                 MLDataType element_type,
                                                                 const OrtAllocatorInfo& location,
                                                                 const TensorShape& shape) {
  auto p_tensor = std::make_unique<Tensor>(element_type, shape, pBuffer, location);
  mlvalue.Init(p_tensor.release(),
               DataTypeImpl::GetType<Tensor>(),
               DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());

  return Status::OK();
}

static Status AllocateTraditionalMLValue(MLValue& mlvalue, const NonTensorTypeBase& type) {
  auto creator = type.GetCreateFunc();
  mlvalue.Init(creator(), &type, type.GetDeleteFunc());
  return Status::OK();
}

// This method is not thread safe!
Status ExecutionFrame::AllocateAsPerAllocationPlan(MLValue& mlvalue, int mlvalue_index, const TensorShape* shape) {
  // if there is a custom allocator for this mlvalue_index, call it to do the allocation
  auto custom_alloc_entry = custom_allocators_.find(mlvalue_index);
  if (custom_alloc_entry != custom_allocators_.cend()) {
    ORT_ENFORCE(shape, "We don't expect custom allocators for non-tensor types, so a shape is mandatory here.");
    return (custom_alloc_entry->second)(*shape, mlvalue);
  }

  const SequentialExecutionPlan* p_seq_exec_plan = session_state_.GetExecutionPlan();
  const auto& alloc_plan = p_seq_exec_plan->allocation_plan;
  ORT_ENFORCE(mlvalue_index >= 0 && mlvalue_index < alloc_plan.size());
  const auto& per_alloc_plan = alloc_plan[mlvalue_index];

  auto alloc_info = per_alloc_plan.location;
  auto ml_type = per_alloc_plan.value_type;
  if (ml_type == nullptr)
    return Status(ONNXRUNTIME, INVALID_ARGUMENT,
                  "Tried to allocate without valid type information, mlvalue index=" + std::to_string(mlvalue_index));

  if (!ml_type->IsTensorType()) {
    return AllocateTraditionalMLValue(mlvalue, *static_cast<const NonTensorTypeBase*>(ml_type));
  }

  ORT_ENFORCE(shape, "Allocation of tensor types requires a shape.");

  // tensors
  auto ml_data_type = static_cast<const TensorTypeBase*>(ml_type)->GetElementType();

  AllocKind alloc_kind = per_alloc_plan.alloc_kind;
  switch (alloc_kind) {
    // Right now for kAllocate and kAllocateOutput we are using same approach.
    // In the future we may want to have different way to handle it.
    case AllocKind::kAllocateOutput:
    case AllocKind::kAllocate: {
      ORT_RETURN_IF_ERROR(AllocateMLValueTensorSelfOwnBuffer(mlvalue, mlvalue_index, ml_data_type, alloc_info, *shape,
                                                             per_alloc_plan.create_fence_if_async));
      break;
    }
    case AllocKind::kReuse: {
      int reuse_mlvalue_index = per_alloc_plan.reused_buffer;
      ORT_RETURN_IF_ERROR(AllocateMLValueTensorPreAllocateBuffer(mlvalue, reuse_mlvalue_index,
                                                                 ml_data_type, alloc_info, *shape,
                                                                 per_alloc_plan.create_fence_if_async));
      break;
    }
    default: {
      std::ostringstream ostr;
      ostr << "Invalid allocation kind: " << static_cast<std::underlying_type<AllocKind>::type>(alloc_kind);
      return Status(ONNXRUNTIME, FAIL, ostr.str());
    }
  }

  return Status::OK();
}

AllocatorPtr ExecutionFrame::GetAllocatorImpl(const OrtAllocatorInfo& info) const {
  return utils::GetAllocator(session_state_, info);
}

// This method is not thread safe!
// Return S_OK and nullptr if index map to an value that is an unused optional input/output
Status ExecutionFrame::CreateNodeOutputMLValueImpl(MLValue& mlvalue, int mlvalue_idx, const TensorShape* shape) {
  return AllocateAsPerAllocationPlan(mlvalue, mlvalue_idx, shape);
}

Status ExecutionFrame::ReleaseMLValueImpl(int mlvalue_idx) {
  ORT_RETURN_IF_ERROR(IExecutionFrame::ReleaseMLValueImpl(mlvalue_idx));
  TraceFree(mlvalue_idx);
  return Status::OK();
}

const AllocPlanPerValue& ExecutionFrame::GetAllocationPlan(int mlvalue_idx) {
  const SequentialExecutionPlan* p_seq_exec_plan = session_state_.GetExecutionPlan();
  const auto& alloc_plan = p_seq_exec_plan->allocation_plan;
  ORT_ENFORCE(mlvalue_idx != NodeIndexInfo::kInvalidEntry && mlvalue_idx < alloc_plan.size());
  return alloc_plan[mlvalue_idx];
}

void ExecutionFrame::TraceAllocate(int mlvalue_idx, size_t size) {
  // don't trace the output tensors.
  auto& allocation_plan = GetAllocationPlan(mlvalue_idx);
  if (planner_ && allocation_plan.alloc_kind != AllocKind::kAllocateOutput) {
    auto status = planner_->TraceAllocation(mlvalue_idx, size);
    if (!status.IsOK())
      LOGS(session_state_.Logger(), WARNING) << "TraceAllocation for mlvalue_idx=" << mlvalue_idx << " size=" << size
                                             << " failed: " << status.ErrorMessage();
  }
}

void ExecutionFrame::TraceFree(int mlvalue_idx) {
  // don't trace free on output tensors.
  if (planner_ && !IsOutput(mlvalue_idx)) {
    const SequentialExecutionPlan* p_seq_exec_plan = session_state_.GetExecutionPlan();
    const auto& alloc_plan = p_seq_exec_plan->allocation_plan;
    const auto& per_alloc_plan = alloc_plan.at(mlvalue_idx);

    // only trace tensors
    auto ml_type = per_alloc_plan.value_type;
    if (ml_type->IsTensorType()) {
      // tensors
      auto ml_data_type = static_cast<const TensorTypeBase*>(ml_type)->GetElementType();
      // don't trace string tensors
      if (ml_data_type != DataTypeImpl::GetType<std::string>()) {
        auto status = planner_->TraceFree(mlvalue_idx);
        if (!status.IsOK()) {
          LOGS(session_state_.Logger(), WARNING) << "TraceFree for mlvalue_idx=" << mlvalue_idx
                                                 << " failed: " << status.ErrorMessage();
        }
      }
    }
  }
}

// generate memory pattern based on the tracing of memory allocation/free in current execution
// return error if the planner is not setup.
Status ExecutionFrame::GeneratePatterns(MemoryPatternGroup* out) const {
  if (!planner_) {
    return Status(ONNXRUNTIME, FAIL, "Memory pattern planner is not enabled on this execution framework.");
  }

  return planner_->GeneratePatterns(out);
}

}  // namespace onnxruntime
