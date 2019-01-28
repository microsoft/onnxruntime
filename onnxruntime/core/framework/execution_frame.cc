// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/execution_frame.h"

#include <sstream>

#include "core/framework/mem_pattern_planner.h"
#include "core/framework/ml_value_patterns_planner.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/framework/utils.h"

using namespace ::onnxruntime::common;
namespace onnxruntime {

ExecutionFrame::ExecutionFrame(const std::unordered_map<std::string, MLValue>& feeds,
                               const std::vector<std::string>& output_names,
                               const std::vector<MLValue>& fetches,
                               const ::onnxruntime::SessionState& session_state)
    : session_state_(session_state), mem_patterns_(nullptr), planner_(nullptr) {
  auto* graph = session_state.GetGraphViewer();
  ORT_ENFORCE(graph);
  Init(*graph, feeds, output_names, fetches);

  // If the session enable memory pattern optimization
  // and we have execution plan generated, try to setup
  // memory pattern optimization.
  if (session_state.GetEnableMemoryPattern() &&
      session_state.GetExecutionPlan()) {
    std::vector<TensorShape> input_shapes;
    bool all_tensors = true;
    for (const auto& feed : feeds) {
      if (!(feed.second.IsTensor())) {
        all_tensors = false;
        break;
      }
      auto& tensor = feed.second.Get<Tensor>();
      input_shapes.push_back(tensor.Shape());
    }
    // if there is some traditional ml value type in inputs
    // disable the memory pattern optimization.
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
          void* buffer = mem_patterns_->patterns[i].PeakSize() > 0 ? alloc->Alloc(mem_patterns_->patterns[i].PeakSize()) : nullptr;
          buffers_[mem_patterns_->locations[i]] = BufferUniquePtr(buffer, alloc);
        }
      }
    }
  }
}

ExecutionFrame::~ExecutionFrame() = default;

Status ExecutionFrame::AllocateMLValueTensorSelfOwnBuffer(int mlvalue_index,
                                                          const DataTypeImpl* element_type,
                                                          const OrtAllocatorInfo& location,
                                                          const TensorShape& shape,
                                                          bool create_fence) {
  ORT_ENFORCE(mlvalue_index >= 0 && static_cast<size_t>(mlvalue_index) < all_values_.size());
  return AllocateMLValueTensorSelfOwnBufferHelper(mlvalue_index, element_type, location, shape, create_fence);
}

Status ExecutionFrame::AllocateMLValueTensorSelfOwnBufferHelper(int mlvalue_index,
                                                                const DataTypeImpl* element_type,
                                                                const OrtAllocatorInfo& location,
                                                                const TensorShape& shape,
                                                                bool create_fence) {
  if (mlvalue_index < 0)
    return Status(ONNXRUNTIME, FAIL, "Trying to allocate memory for unused optional inputs/outputs");

  auto p_mlvalue = &all_values_[mlvalue_index];
  if (p_mlvalue->IsAllocated()) {
    return Status::OK();
  }
  auto alloc = GetAllocator(location);
  size_t size;
  {
    int64_t len = shape.Size();
    if (len < 0) {
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Tensor shape cannot contain any negative value");
    }
    if (!IAllocator::CalcMemSizeForArrayWithAlignment<64>(len, element_type->Size(), &size)) {
      return Status(ONNXRUNTIME, FAIL, "size overflow");
    }
  }
  // create fence if needed
  if (create_fence) {
    ORT_ENFORCE(p_mlvalue->Fence() == nullptr);
    FencePtr f = alloc->CreateFence(&SessionState());
    // it is OK to have fence been nullptr if the execution provider has no async execution,
    // and allocator::CreateFence returns nullptr
    p_mlvalue->SetFence(f);
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
              p_mlvalue, static_cast<void*>(static_cast<char*>(buffer) + block->offset_),
              element_type, location, shape);
          return status;
        }
        if (block->size_ != size) {
          LOGS_DEFAULT(WARNING) << "For mlvalue with index: " << mlvalue_index << ", block in memory pattern size is: "
                                << block->size_ << " but the actually size is: " << size << ", fall back to default allocation behavior";
        } else if (it == buffers_.end()) {
          LOGS_DEFAULT(WARNING) << "For mlvalue with index: " << mlvalue_index << ", block not found in target loation. "
                                                                                  " fall back to default allocation behavior";
        }
      }
    }
  }
  //no memory pattern, or the pattern is not correct.
  void* buffer = size == 0 ? nullptr : alloc->Alloc(size);
  std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(element_type,
                                                              shape,
                                                              buffer,
                                                              location,
                                                              alloc);

  p_mlvalue->Init(p_tensor.release(),
                  DataTypeImpl::GetType<Tensor>(),
                  DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());

  // trace the memory allocation.
  // don't trace the memory allocation on string tensors, as it need
  // placement new, we don't support it in memory pattern optimization.
  if (element_type != DataTypeImpl::GetType<std::string>())
    TraceAllocate(mlvalue_index, size);

  return Status::OK();
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

Status ExecutionFrame::AllocateTensorWithSelfOwnBuffer(const int index,
                                                       const DataTypeImpl* element_type,
                                                       const OrtAllocatorInfo& location,
                                                       const TensorShape& shape,
                                                       bool create_fence) {
  ORT_ENFORCE(index >= 0 && static_cast<size_t>(index) < node_values_.size());
  return AllocateMLValueTensorSelfOwnBufferHelper(node_values_[index], element_type, location, shape, create_fence);
}

Status ExecutionFrame::AllocateMLValueTensorPreAllocateBuffer(int mlvalue_index_to_allocate,
                                                              int mlvalue_index_reuse,
                                                              const DataTypeImpl* element_type,
                                                              const OrtAllocatorInfo& location,
                                                              const TensorShape& shape,
                                                              bool create_fence) {
  ORT_ENFORCE(mlvalue_index_to_allocate >= 0 && mlvalue_index_to_allocate < all_values_.size());
  MLValue* p_mlvalue = &all_values_[mlvalue_index_to_allocate];

  ORT_ENFORCE(mlvalue_index_reuse >= 0 && mlvalue_index_reuse < all_values_.size());
  MLValue* p_mlvalue_reuse = &all_values_[mlvalue_index_reuse];

  auto* reuse_tensor = p_mlvalue_reuse->GetMutable<Tensor>();
  void* reuse_buffer = reuse_tensor->MutableDataRaw();

  // create fence on reused mlvalue if needed
  // TODO: differentiate reuse and alias, by add AllocKind::kAlias?
  if (create_fence && p_mlvalue_reuse->Fence() == nullptr) {
    FencePtr f = GetAllocator(location)->CreateFence(&SessionState());
    p_mlvalue_reuse->SetFence(f);
  }

  // reused MLValue share the same fence
  p_mlvalue->ShareFenceWith(*p_mlvalue_reuse);
  return AllocateTensorWithPreAllocateBufferHelper(p_mlvalue, reuse_buffer, element_type, location, shape);
}

Status ExecutionFrame::AllocateTensorWithPreAllocateBufferHelper(MLValue* p_mlvalue,
                                                                 void* pBuffer,
                                                                 const DataTypeImpl* element_type,
                                                                 const OrtAllocatorInfo& location,
                                                                 const TensorShape& shape) {
  if (p_mlvalue->IsAllocated()) {
    return Status::OK();
  }
  std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(element_type,
                                                              shape,
                                                              pBuffer,
                                                              location);
  p_mlvalue->Init(p_tensor.release(),
                  DataTypeImpl::GetType<Tensor>(),
                  DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());

  return Status::OK();
}

Status ExecutionFrame::AllocateTensorWithPreAllocateBuffer(const int offset,
                                                           void* pBuffer,
                                                           const DataTypeImpl* element_type,
                                                           const OrtAllocatorInfo& location,
                                                           const TensorShape& shape) {
  ORT_ENFORCE(offset >= 0 && offset < node_values_.size());
  if (node_values_[offset] < 0)
    return Status(ONNXRUNTIME, FAIL, "Trying to allocate memory for unused optional inputs/outputs");
  auto value = &all_values_[node_values_[offset]];
  return AllocateTensorWithPreAllocateBufferHelper(value, pBuffer, element_type, location, shape);
}

void ExecutionFrame::Release(const int offset) {
  ORT_ENFORCE(offset >= 0 && offset < node_offsets_.size());
  if (node_values_[offset] >= 0 && node_values_[offset] < all_values_.size()) {
    all_values_[node_values_[offset]] = MLValue();
    TraceFree(node_values_[offset]);
  }
}

Status AllocateTraditionalMLValue(MLValue* p_mlvalue,
                                  const NonTensorTypeBase* type,
                                  const MLValueAllocationParameters& parameters) {
  // right now we don't need any parameter for ml value creation,
  // keep it in api for extensibility
  ORT_UNUSED_PARAMETER(parameters);
  auto creator = type->GetCreateFunc();
  p_mlvalue->Init(creator(),
                  type,
                  type->GetDeleteFunc());
  return Status::OK();
}

// This method is not thread safe!
Status ExecutionFrame::AllocateAsPerAllocationPlan(int mlvalue_index,
                                                   const MLValueAllocationParameters& parameters) {
  if (mlvalue_index < 0 || mlvalue_index >= all_values_.size())
    return Status(ONNXRUNTIME, INVALID_ARGUMENT,
                  "Tried to allocated with invalid mlvalue index: " + std::to_string(mlvalue_index));
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
    return AllocateTraditionalMLValue(&all_values_[mlvalue_index],
                                      static_cast<const NonTensorTypeBase*>(ml_type),
                                      parameters);
  }

  // tensors
  auto ml_data_type = static_cast<const TensorTypeBase*>(ml_type)->GetElementType();

  AllocKind alloc_kind = per_alloc_plan.alloc_kind;
  switch (alloc_kind) {
    // Right now for kAllocate and kAllocateOutput we are using same approach.
    // In the future we may want to have different way to handle it.
    case AllocKind::kAllocateOutput:
    case AllocKind::kAllocate: {
      ORT_RETURN_IF_ERROR(AllocateMLValueTensorSelfOwnBuffer(mlvalue_index,
                                                                     ml_data_type,
                                                                     alloc_info,
                                                                     parameters.GetTensorShape(),
                                                                     per_alloc_plan.create_fence_if_async));
      break;
    }
    case AllocKind::kReuse: {
      int reuse_mlvalue_index = per_alloc_plan.reused_buffer;
      ORT_RETURN_IF_ERROR(AllocateMLValueTensorPreAllocateBuffer(mlvalue_index,
                                                                         reuse_mlvalue_index,
                                                                         ml_data_type,
                                                                         alloc_info,
                                                                         parameters.GetTensorShape(),
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

void ExecutionFrame::Init(const onnxruntime::GraphViewer& graph,
                          const std::unordered_map<std::string, MLValue>& feeds,
                          const std::vector<std::string>& output_names,
                          const std::vector<MLValue>& fetches) {
  // 1. resize the node_offsets and all_value_ vector
  // We need to use the max index rather than number of nodes as we use Node.Index()
  // when inserting into node_offsets_
  auto max_node_index = graph.MaxNodeIndex();
  node_offsets_.resize(max_node_index);

  auto& mlvalue_idx_map = session_state_.GetMLValueNameIdxMap();

  all_values_.resize(mlvalue_idx_map.MaxIdx() + 1);

  // 2. handle the weights.
  for (const auto& entry : session_state_.GetInitializedTensors()) {
    auto mlvalue_index = entry.first;
    all_values_[mlvalue_index] = entry.second;  // this copy should be cheap
  }

  // 3. handle feed in values
  for (const auto& feed : feeds) {
    int mlvalue_idx;
    Status status = mlvalue_idx_map.GetIdx(feed.first, mlvalue_idx);
    ORT_ENFORCE(status.IsOK(), status.ErrorMessage());
    // we are sharing the underline tensor/object for MLValue
    all_values_[mlvalue_idx] = feed.second;
  }

  // 4. Handle non-empty output vector
  if (!fetches.empty()) {
    // should've already verified this much before when Run() starts
    ORT_ENFORCE(output_names.size() == fetches.size(),
                        "output_names vector size: " + std::to_string(output_names.size()) +
                            " does not match that of fetches vector: " + std::to_string(fetches.size()));

    // setup output_indices_, we dont' want to generate mem plan on output tensors.
    output_indices_.reserve(output_names.size());
    auto idx = 0;
    for (const auto& oname : output_names) {
      int mlvalue_idx;
      Status status = mlvalue_idx_map.GetIdx(oname, mlvalue_idx);
      ORT_ENFORCE(status.IsOK(), status.ErrorMessage());
      all_values_[mlvalue_idx] = fetches.at(idx++);
      output_indices_.push_back(mlvalue_idx);
    }
  }

  // 5. set node args
  std::size_t total_def_count{};
  for (const auto& node : graph.Nodes())
  {
    node.ForEachDef([&](const onnxruntime::NodeArg& /*arg*/, bool /*is_input*/) {
      ++total_def_count;
    });
  }
  node_values_.reserve(total_def_count);

  for (auto& node : graph.Nodes()) {
    ORT_ENFORCE(node.Index() < node_offsets_.size());
    node_offsets_[node.Index()] = static_cast<int>(node_values_.size());

    for (auto input_def : node.InputDefs()) {
      SetupNodeArg(input_def);
    }

    for (auto input_def : node.ImplicitInputDefs()) {
      SetupNodeArg(input_def);
    }

    for (auto output_def : node.OutputDefs()) {
      SetupNodeArg(output_def);
    }
  }
}

void ExecutionFrame::SetupNodeArg(const onnxruntime::NodeArg* arg) {
  ORT_ENFORCE(arg);
  auto& name = arg->Name();
  //if the arg's name is empty, it is an not needed optional input/output
  //set index to -1
  if (name.empty()) {
    node_values_.push_back(-1);
  } else {
    int index;
    Status status = session_state_.GetMLValueNameIdxMap().GetIdx(name, index);
    ORT_ENFORCE(status.IsOK(), status.ErrorMessage());
    node_values_.push_back(index);
  }
}

void ExecutionFrame::TraceFree(int mlvalue_idx) {
  // don't trace free on output tensors.
  if (planner_ &&
      std::find(output_indices_.begin(), output_indices_.end(), mlvalue_idx) == output_indices_.end()) {
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

// Return nullptr if index map to an value that is an unused optional input/output
const MLValue* ExecutionFrame::GetNodeInputOrOutputMLValue(int index) const {
  ORT_ENFORCE(index >= 0 && static_cast<size_t>(index) < node_values_.size());
  return node_values_[index] >= 0 ? &all_values_[node_values_[index]] : nullptr;
}

// Return nullptr if index map to an value that is an unused optional input/output
MLValue* ExecutionFrame::GetMutableNodeInputOrOutputMLValue(int index) {
  return const_cast<MLValue*>(GetNodeInputOrOutputMLValue(index));
}

AllocatorPtr ExecutionFrame::GetAllocator(const OrtAllocatorInfo& info) {
  return utils::GetAllocator(session_state_, info);
}

static inline void VerifyShape(const MLValue* p_mlvalue,
                               const MLValueAllocationParameters& parameters) {
  if (p_mlvalue->IsTensor()) {
    const Tensor* tensor = &p_mlvalue->Get<Tensor>();

    ORT_ENFORCE(tensor->Shape() == parameters.GetTensorShape(),
                        "MLValue shape verification failed. Current shape:", tensor->Shape(),
                        " Requested shape:", parameters.GetTensorShape());
  }
}

// This method is not thread safe!
// Return S_OK and nullptr if index map to an value that is an unused optional input/output
Status ExecutionFrame::GetOrCreateNodeOutputMLValue(int index,
                                                    const MLValueAllocationParameters& parameters,
                                                    MLValue*& p_mlvalue) {
  if (index < 0 || static_cast<size_t>(index) >= node_values_.size()) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "Try to access with invalid node value index: " + std::to_string(index));
  }

  // return nullptr if it is optional
  if (node_values_[index] < 0) {
    p_mlvalue = nullptr;
    return Status::OK();
  }

  p_mlvalue = &all_values_.at(node_values_[index]);

  if (p_mlvalue->IsAllocated()) {
    // The ml has already been allocated.
    // Now only tensor need to be check.
    VerifyShape(p_mlvalue, parameters);  // TODO find a better way to do this
    return Status::OK();
  }
    // It's not allocated, then allocate it with given shape and return.
    // Perform allocation based on the allocation plan
    ORT_RETURN_IF_ERROR(AllocateAsPerAllocationPlan(node_values_[index], parameters));
    return Status::OK();
}

Status ExecutionFrame::ReleaseMLValue(int mlvalue_idx) {
  if (mlvalue_idx < 0 || static_cast<size_t>(mlvalue_idx) >= all_values_.size()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "invalid index ", mlvalue_idx);
  }
  all_values_[mlvalue_idx] = MLValue();
  TraceFree(mlvalue_idx);
  return Status::OK();
}

const SequentialExecutionPlan::AllocPlanPerValue& ExecutionFrame::GetAllocationPlan(int mlvalue_idx) {
  const SequentialExecutionPlan* p_seq_exec_plan = session_state_.GetExecutionPlan();
  const auto& alloc_plan = p_seq_exec_plan->allocation_plan;
  ORT_ENFORCE(mlvalue_idx >= 0 && mlvalue_idx < alloc_plan.size());
  return alloc_plan[mlvalue_idx];
}
}  // namespace onnxruntime
