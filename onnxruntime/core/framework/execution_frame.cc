// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/execution_frame.h"

#include <sstream>

#include "core/framework/mem_pattern_planner.h"
#include "core/framework/execution_plan_base.h"
#include "core/framework/sequential_execution_plan.h"
#include "core/framework/ort_value_pattern_planner.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/node_index_info.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/framework/utils.h"

using namespace onnxruntime::common;

namespace onnxruntime {

IExecutionFrame::IExecutionFrame(const std::vector<int>& feed_mlvalue_idxs, const std::vector<OrtValue>& feeds,
                                 const std::unordered_map<int, OrtValue>& initializers,
                                 const std::vector<int>& fetch_mlvalue_idxs, const std::vector<OrtValue>& fetches,
                                 const OrtValueNameIdxMap& ort_value_idx_map, const NodeIndexInfo& node_index_info)
    : node_index_info_(node_index_info),
      all_values_size_(static_cast<size_t>(ort_value_idx_map.MaxIdx()) + 1),
      fetch_mlvalue_idxs_(fetch_mlvalue_idxs) {
  ORT_ENFORCE(feeds.size() == feed_mlvalue_idxs.size());
  ORT_ENFORCE(fetches.empty() || fetches.size() == fetch_mlvalue_idxs_.size());
  ORT_ENFORCE(node_index_info_.GetMaxMLValueIdx() == ort_value_idx_map.MaxIdx(),
              "node_index_info and ort_value_idx_map are out of sync and cannot be used");

  Init(feed_mlvalue_idxs, feeds, initializers, fetches);
}

IExecutionFrame::~IExecutionFrame() = default;

// Return nullptr if index map to an value that is an unused optional input/output
const OrtValue* IExecutionFrame::GetNodeInputOrOutputMLValue(int index) const {
  int ort_value_idx = GetNodeIdxToMLValueIdx(index);
  return ort_value_idx != NodeIndexInfo::kInvalidEntry ? &all_values_[ort_value_idx] : nullptr;
}

OrtValue* IExecutionFrame::GetMutableNodeInputOrOutputMLValue(int index) {
  return const_cast<OrtValue*>(GetNodeInputOrOutputMLValue(index));
}

// TO DO: make it thread safe
// This method is not thread safe!
// Return S_OK and nullptr if index map to an value that is an unused optional input/output

Status IExecutionFrame::GetOrCreateNodeOutputMLValue(int index, const TensorShape* shape, OrtValue*& p_ort_value,
                                                     size_t nnz) {
  auto status = Status::OK();
  int ort_value_idx = GetNodeIdxToMLValueIdx(index);

  // return nullptr if it is optional
  if (ort_value_idx == NodeIndexInfo::kInvalidEntry) {
    p_ort_value = nullptr;
  } else {
    p_ort_value = &all_values_[ort_value_idx];

    if (p_ort_value->IsAllocated()) {
      // already allocated. verify shape matches if tensor.
      if (p_ort_value->IsTensor()) {
        const Tensor& tensor = p_ort_value->Get<Tensor>();
        ORT_ENFORCE(shape && tensor.Shape() == *shape,
                    "OrtValue shape verification failed. Current shape:", tensor.Shape(),
                    " Requested shape:", shape ? shape->ToString() : "null");
      }
    } else {
      status = CreateNodeOutputMLValueImpl(*p_ort_value, ort_value_idx, shape, nnz);
    }
  }

  return status;
}

AllocatorPtr IExecutionFrame::GetAllocator(const OrtMemoryInfo& info) const {
  return GetAllocatorImpl(info);
}

Status IExecutionFrame::ReleaseMLValue(int ort_value_idx) { return ReleaseMLValueImpl(ort_value_idx); }

Status IExecutionFrame::ReleaseMLValueImpl(int ort_value_idx) {
  if (ort_value_idx == NodeIndexInfo::kInvalidEntry || static_cast<size_t>(ort_value_idx) >= all_values_size_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "invalid index ", ort_value_idx);
  }

  // If fence is available, check whether async read has completed or not.
  Fence_t fence = GetMLValue(ort_value_idx).Fence();
  if (fence && !fence->CanRelease()) {
    // Async data reading is not done yet, defer mem release until Session.run() end.
    return Status::OK();
  }

  all_values_[ort_value_idx] = OrtValue();
  return Status::OK();
}

int IExecutionFrame::GetNodeIdxToMLValueIdx(int index) const {
  // the validity of index is checked by GetMLValueIndex
  int ort_value_idx = node_index_info_.GetMLValueIndex(index);
  return ort_value_idx;
}

void IExecutionFrame::Init(const std::vector<int>& feed_mlvalue_idxs, const std::vector<OrtValue>& feeds,
                           const std::unordered_map<int, OrtValue>& initializers,
                           const std::vector<OrtValue>& fetches) {
  // 1. resize the all_value_ vector
  all_values_.resize(all_values_size_);

  // 2. Handle non-empty output vector
  if (!fetches.empty()) {
    auto num_fetches = fetch_mlvalue_idxs_.size();

    for (size_t idx = 0; idx < num_fetches; ++idx) {
      int ort_value_idx = fetch_mlvalue_idxs_[idx];
      all_values_[ort_value_idx] = fetches[idx];
    }
  }

  // 3. handle the weights.
  // We do this after the fetches to handle an edge case (possibly dubious) where a Constant is an output.
  // The Constant gets lifted to an initializer so there's no Node producing the value as an output during Graph
  // execution (i.e. Graph execution won't write the value to all_values_).
  // A non-empty fetches vector will overwrite the actual weight in all_values_[ort_value_idx] if we did this earlier.
  // This makes the ONNX Constant test (onnx\backend\test\data\node\test_constant) happy as that
  // involves a graph with a single Constant node.
  for (const auto& entry : initializers) {
    int ort_value_index = entry.first;
    all_values_[ort_value_index] = entry.second;
  }

  // 4. handle feed in values. these can override initializer values so must be last
  for (size_t idx = 0, end = feed_mlvalue_idxs.size(); idx < end; ++idx) {
    int ort_value_idx = feed_mlvalue_idxs[idx];
    // we are sharing the underline tensor/object for MLValue
    all_values_[ort_value_idx] = feeds[idx];
  }
}

Status IExecutionFrame::GetOutputs(std::vector<OrtValue>& fetches) {
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

bool IExecutionFrame::IsOutput(int ort_value_idx) const {
  return std::find(fetch_mlvalue_idxs_.begin(), fetch_mlvalue_idxs_.end(), ort_value_idx) != fetch_mlvalue_idxs_.end();
}

ExecutionFrame::ExecutionFrame(const std::vector<int>& feed_mlvalue_idxs, const std::vector<OrtValue>& feeds,
                               const std::vector<int>& fetch_mlvalue_idxs, const std::vector<OrtValue>& fetches,
                               const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                               const SessionState& session_state)
    : IExecutionFrame(feed_mlvalue_idxs, feeds, session_state.GetInitializedTensors(), fetch_mlvalue_idxs, fetches,
                      session_state.GetOrtValueNameIdxMap(), session_state.GetNodeIndexInfo()),
      session_state_(session_state),
      mem_patterns_(nullptr),
      planner_(nullptr) {
  // map the custom allocators to ort_value_idx entries
  if (!fetch_allocators.empty()) {
    for (size_t idx = 0, end = fetch_mlvalue_idxs.size(); idx < end; ++idx) {
      int ort_value_idx = fetch_mlvalue_idxs[idx];

      auto custom_alloc_entry = fetch_allocators.find(idx);
      if (custom_alloc_entry != fetch_allocators.cend()) {
        custom_allocators_[ort_value_idx] = custom_alloc_entry->second;
      }
    }
  }

  // If the session enable memory pattern optimization
  // and we have execution plan generated, try to setup
  // memory pattern optimization.
  if (session_state.GetEnableMemoryPattern() && session_state.GetExecutionPlan()) {
    std::vector<std::reference_wrapper<const TensorShape>> input_shapes;
    bool all_tensors = true;
    // Reserve mem to avoid re-allocation.
    input_shapes.reserve(feeds.size());
    for (const auto& feed : feeds) {
      if (!(feed.IsTensor())) {
        all_tensors = false;
        break;
      }
      auto& tensor = feed.Get<Tensor>();
      input_shapes.push_back(std::cref(tensor.Shape()));
    }

    //if there are some traditional ml value type in inputs disable the memory pattern optimization.
    if (all_tensors) {
      mem_patterns_ = session_state.GetMemoryPatternGroup(input_shapes);
      // if no existing patterns, generate one in this executionframe
      if (!mem_patterns_) {
        planner_ = onnxruntime::make_unique<OrtValuePatternPlanner>(*session_state.GetExecutionPlan());
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

Status ExecutionFrame::AllocateMLValueTensorSelfOwnBuffer(OrtValue& ort_value, int ort_value_index,
                                                          MLDataType element_type, const OrtMemoryInfo& location,
                                                          const TensorShape& shape, bool create_fence) {
  return AllocateMLValueTensorSelfOwnBufferHelper(ort_value, ort_value_index, element_type, location, shape,
                                                  create_fence);
}

Status ExecutionFrame::AllocateMLValueTensorSelfOwnBufferHelper(OrtValue& ort_value, int ort_value_index,
                                                                MLDataType element_type,
                                                                const OrtMemoryInfo& location,
                                                                const TensorShape& shape, bool create_fence) {
  if (ort_value_index == NodeIndexInfo::kInvalidEntry) {
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
    ORT_ENFORCE(ort_value.Fence() == nullptr);
    FencePtr f = alloc->CreateFence(&session_state_);
    // it is OK to have fence been nullptr if the execution provider has no async execution,
    // and allocator::CreateFence returns nullptr
    ort_value.SetFence(f);
  }

  // if we have pre-calculated memory pattern, and the ort_value is not output mlvalue
  // try to allocated on pre-allocated big chunk.
  const auto& per_alloc_plan = GetAllocationPlan(ort_value_index);
  if (mem_patterns_ && per_alloc_plan.alloc_kind != AllocKind::kAllocateOutput) {
    auto pattern = mem_patterns_->GetPatterns(location);
    if (pattern) {
      auto block = pattern->GetBlock(ort_value_index);
      // if block not found, fall back to default behavior
      if (block) {
        auto it = buffers_.find(location);
        // if the block is not correct, log message then fall back to default behavior
        if (it != buffers_.end() && block->size_ == size) {
          void* buffer = it->second.get();
          auto status = AllocateTensorWithPreAllocateBufferHelper(
              ort_value, static_cast<void*>(static_cast<char*>(buffer) + block->offset_), element_type, location,
              shape);
          return status;
        }
        if (block->size_ != size) {
          // the block size may vary especially if the model has NonZero ops, or different sequence lengths are
          // fed in, so use VERBOSE as the log level as it's expected.
          // TODO: Should we re-use the block if the size is large enough? Would probably need to allow it
          // to be freed if the size difference was too large so our memory usage doesn't stick at a high water mark
          LOGS_DEFAULT(VERBOSE) << "For ort_value with index: " << ort_value_index
                                << ", block in memory pattern size is: " << block->size_
                                << " but the actually size is: " << size
                                << ", fall back to default allocation behavior";
        } else if (it == buffers_.end()) {
          LOGS_DEFAULT(WARNING) << "For ort_value with index: " << ort_value_index
                                << ", block not found in target location. fall back to default allocation behavior";
        }
      }
    }
  }
  //no memory pattern, or the pattern is not correct.
  std::unique_ptr<Tensor> p_tensor = onnxruntime::make_unique<Tensor>(element_type, shape, alloc);

  {
    auto ml_tensor = DataTypeImpl::GetType<Tensor>();
    ort_value.Init(p_tensor.release(), ml_tensor, ml_tensor->GetDeleteFunc());
  }

  // trace the memory allocation.
  // don't trace the memory allocation on string tensors, as it need
  // placement new, we don't support it in memory pattern optimization.
  if (!utils::IsDataTypeString(element_type)) {
    TraceAllocate(ort_value_index, size);
  }

  return Status::OK();
}

Status ExecutionFrame::AllocateMLValueTensorPreAllocateBuffer(OrtValue& ort_value, int ort_value_index_reuse,
                                                              MLDataType element_type, const OrtMemoryInfo& location,
                                                              const TensorShape& shape, bool create_fence) {
  OrtValue& ort_value_reuse = GetMutableMLValue(ort_value_index_reuse);

  auto* reuse_tensor = ort_value_reuse.GetMutable<Tensor>();
  auto buffer_num_elements = reuse_tensor->Shape().Size();
  auto required_num_elements = shape.Size();

  // check number of elements matches. shape may not be an exact match (e.g. Reshape op)
  if (buffer_num_elements != required_num_elements) {
    // could be an allocation planner bug (less likely) or the model incorrectly uses something like 'None'
    // as a dim_param, or -1 in dim_value in multiple places making the planner think those shapes are equal.
    auto message = onnxruntime::MakeString(
        "Shape mismatch attempting to re-use buffer. ",
        reuse_tensor->Shape(), " != ", shape,
        ". Validate usage of dim_value (values should be > 0) and "
        "dim_param (all values with the same string should equate to the same size) in shapes in the model.");

    // be generous and use the buffer if it's large enough. log a warning though as it indicates a bad model
    if (buffer_num_elements >= required_num_elements) {
      LOGS_DEFAULT(WARNING) << message;
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, message);
    }
  }

  void* reuse_buffer = reuse_tensor->MutableDataRaw();

  // create fence on reused ort_value if needed
  // TODO: differentiate reuse and alias, by add AllocKind::kAlias?
  if (create_fence && ort_value_reuse.Fence() == nullptr) {
    FencePtr f = GetAllocator(location)->CreateFence(&session_state_);
    ort_value_reuse.SetFence(f);
  }

  // reused OrtValue share the same fence
  ort_value.ShareFenceWith(ort_value_reuse);
  return AllocateTensorWithPreAllocateBufferHelper(ort_value, reuse_buffer, element_type, location, shape);
}

Status ExecutionFrame::AllocateTensorWithPreAllocateBufferHelper(OrtValue& ort_value, void* pBuffer,
                                                                 MLDataType element_type,
                                                                 const OrtMemoryInfo& location,
                                                                 const TensorShape& shape) {
  auto ml_tensor = DataTypeImpl::GetType<Tensor>();
  auto p_tensor = onnxruntime::make_unique<Tensor>(element_type, shape, pBuffer, location);
  ort_value.Init(p_tensor.release(), ml_tensor, ml_tensor->GetDeleteFunc());

  return Status::OK();
}

static Status AllocateTraditionalMLValue(OrtValue& ort_value, const NonTensorTypeBase& type) {
  auto creator = type.GetCreateFunc();
  ort_value.Init(creator(), &type, type.GetDeleteFunc());
  return Status::OK();
}

static Status AllocateSparseTensor(MLValue& mlvalue, const DataTypeImpl& ml_type, AllocatorPtr allocator,
                                   const TensorShape& shape, size_t nnz, bool create_fence,
                                   const SessionState& session_state) {
  auto element_type = ml_type.AsSparseTensorType()->GetElementType();
  auto sparse = onnxruntime::make_unique<SparseTensor>(element_type, shape, nnz, allocator);
  auto deleter = DataTypeImpl::GetType<SparseTensor>()->GetDeleteFunc();
  mlvalue.Init(sparse.release(), DataTypeImpl::GetType<SparseTensor>(), deleter);

  // create fence if needed
  if (create_fence) {
    ORT_ENFORCE(mlvalue.Fence() == nullptr);
    FencePtr f = allocator->CreateFence(&session_state);
    mlvalue.SetFence(f);
  }

  return Status::OK();
}

// This method is not thread safe!
Status ExecutionFrame::AllocateAsPerAllocationPlan(OrtValue& ort_value, int ort_value_index, const TensorShape* shape,
                                                   size_t nnz) {
  const SequentialExecutionPlan* p_seq_exec_plan = session_state_.GetExecutionPlan();
  const auto& alloc_plan = p_seq_exec_plan->allocation_plan;
  ORT_ENFORCE(ort_value_index >= 0 && static_cast<size_t>(ort_value_index) < alloc_plan.size());
  const auto& per_alloc_plan = alloc_plan[ort_value_index];

  const auto& alloc_info = per_alloc_plan.location;
  const auto* ml_type = per_alloc_plan.value_type;
  if (ml_type == nullptr) {
    return Status(
        ONNXRUNTIME, INVALID_ARGUMENT,
        "Tried to allocate without valid type information, ort_value index=" + std::to_string(ort_value_index));
  }

  // if there is a custom allocator for this ort_value_index, call it to do the allocation
  auto custom_alloc_entry = custom_allocators_.find(ort_value_index);
  if (custom_alloc_entry != custom_allocators_.cend()) {
    ORT_ENFORCE(shape, "We don't expect custom allocators for non-tensor types, so a shape is mandatory here.");
    bool allocated = false;
    // see if custom allocator can handle allocation
    auto status = (custom_alloc_entry->second)(*shape, alloc_info, ort_value, allocated);
    if (allocated || !status.IsOK())
      return status;
  }

  if (ml_type->IsTensorType()) {
    ORT_ENFORCE(shape, "Allocation of tensor types requires a shape.");

    // tensors
    const auto* ml_data_type = static_cast<const TensorTypeBase*>(ml_type)->GetElementType();

    AllocKind alloc_kind = per_alloc_plan.alloc_kind;
    switch (alloc_kind) {
      // Right now for kAllocate and kAllocateOutput we are using same approach.
      // In the future we may want to have different way to handle it.
      case AllocKind::kAllocateOutput:
      case AllocKind::kAllocate: {
        ORT_RETURN_IF_ERROR(AllocateMLValueTensorSelfOwnBuffer(ort_value, ort_value_index, ml_data_type, alloc_info,
                                                               *shape, per_alloc_plan.create_fence_if_async));
        break;
      }
      case AllocKind::kReuse: {
        int reuse_mlvalue_index = per_alloc_plan.reused_buffer;
        ORT_RETURN_IF_ERROR(AllocateMLValueTensorPreAllocateBuffer(
            ort_value, reuse_mlvalue_index, ml_data_type, alloc_info, *shape, per_alloc_plan.create_fence_if_async));
        break;
      }
      case AllocKind::kShare: {
        int reuse_mlvalue_index = per_alloc_plan.reused_buffer;
        // copy at the OrtValue level so the shared_ptr for the data is shared between the two OrtValue instances
        ort_value = GetMutableMLValue(reuse_mlvalue_index);
        break;
      }
      default: {
        std::ostringstream ostr;
        ostr << "Invalid allocation kind: " << static_cast<std::underlying_type<AllocKind>::type>(alloc_kind);
        return Status(ONNXRUNTIME, FAIL, ostr.str());
      }
    }

    return Status::OK();
  } else if (ml_type->IsSparseTensorType()) {
    return AllocateSparseTensor(ort_value, *ml_type, GetAllocator(alloc_info),
                                *shape, nnz, per_alloc_plan.create_fence_if_async, session_state_);
  } else {
    return AllocateTraditionalMLValue(ort_value, *static_cast<const NonTensorTypeBase*>(ml_type));
  }
}

AllocatorPtr ExecutionFrame::GetAllocatorImpl(const OrtMemoryInfo& info) const {
  return utils::GetAllocator(session_state_, info);
}

// This method is not thread safe!
// Return S_OK and nullptr if index map to an value that is an unused optional input/output
Status ExecutionFrame::CreateNodeOutputMLValueImpl(OrtValue& ort_value, int ort_value_idx, const TensorShape* shape, size_t nnz) {
  return AllocateAsPerAllocationPlan(ort_value, ort_value_idx, shape, nnz);
}

Status ExecutionFrame::ReleaseMLValueImpl(int ort_value_idx) {
  ORT_RETURN_IF_ERROR(IExecutionFrame::ReleaseMLValueImpl(ort_value_idx));
  TraceFree(ort_value_idx);
  return Status::OK();
}

const AllocPlanPerValue& ExecutionFrame::GetAllocationPlan(int ort_value_idx) {
  const SequentialExecutionPlan* p_seq_exec_plan = session_state_.GetExecutionPlan();
  const auto& alloc_plan = p_seq_exec_plan->allocation_plan;
  ORT_ENFORCE(ort_value_idx >= 0 && static_cast<size_t>(ort_value_idx) < alloc_plan.size());
  return alloc_plan[ort_value_idx];
}

void ExecutionFrame::TraceAllocate(int ort_value_idx, size_t size) {
  if (planner_) {
    // don't trace the output tensors.
    auto& allocation_plan = GetAllocationPlan(ort_value_idx);
    if (allocation_plan.alloc_kind == AllocKind::kAllocateOutput) return;
    auto status = planner_->TraceAllocation(ort_value_idx, size);
    if (!status.IsOK())
      LOGS(session_state_.Logger(), WARNING) << "TraceAllocation for ort_value_idx=" << ort_value_idx
                                             << " size=" << size << " failed: " << status.ErrorMessage();
  }
}

void ExecutionFrame::TraceFree(int ort_value_idx) {
  // don't trace free on output tensors.
  if (planner_ && !IsOutput(ort_value_idx)) {
    const SequentialExecutionPlan* p_seq_exec_plan = session_state_.GetExecutionPlan();
    const auto& alloc_plan = p_seq_exec_plan->allocation_plan;
    ORT_ENFORCE(ort_value_idx >= 0 && static_cast<size_t>(ort_value_idx) < alloc_plan.size());
    const auto& per_alloc_plan = alloc_plan[ort_value_idx];

    // only trace tensors
    auto ml_type = per_alloc_plan.value_type;
    if (ml_type->IsTensorType()) {
      // tensors
      auto ml_data_type = static_cast<const TensorTypeBase*>(ml_type)->GetElementType();
      // don't trace string tensors
      if (!utils::IsDataTypeString(ml_data_type)) {
        auto status = planner_->TraceFree(ort_value_idx);
        if (!status.IsOK()) {
          LOGS(session_state_.Logger(), WARNING)
              << "TraceFree for ort_value_idx=" << ort_value_idx << " failed: " << status.ErrorMessage();
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
