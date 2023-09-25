// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/execution_frame.h"

#include <sstream>

#include "core/framework/mem_pattern_planner.h"
#include "core/framework/execution_plan_base.h"
#include "core/framework/sequential_execution_plan.h"
#include "core/framework/ort_value_pattern_planner.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/mldata_type_utils.h"
#include "core/framework/sparse_utils.h"
#include "core/framework/node_index_info.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/framework/TensorSeq.h"
#include "core/framework/utils.h"
#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
#include "core/framework/memory_info.h"
#endif

#include "core/framework/bfc_arena.h"

using namespace onnxruntime::common;

namespace onnxruntime {
#ifdef ORT_ENABLE_STREAM
static StreamAwareArena* AsStreamBasedAllocator(AllocatorPtr allocator) {
  ORT_ENFORCE(allocator.get() != nullptr, "allocator is nullptr");
  if (allocator->Info().alloc_type == OrtArenaAllocator) {
    BFCArena* arena_ptr = static_cast<BFCArena*>(allocator.get());
    return StreamAwareArena::FromBFCArena(*arena_ptr);
  }
  return nullptr;
}
#endif

IExecutionFrame::IExecutionFrame(const OrtValueNameIdxMap& ort_value_idx_map,
                                 const NodeIndexInfo& node_index_info,
                                 gsl::span<const int> fetch_mlvalue_idxs)
    : node_index_info_(node_index_info),
      all_values_size_(static_cast<size_t>(ort_value_idx_map.MaxIdx()) + 1),
      fetch_mlvalue_idxs_(fetch_mlvalue_idxs.begin(), fetch_mlvalue_idxs.end()),
      ort_value_idx_map_(ort_value_idx_map) {
  ORT_ENFORCE(node_index_info_.GetMaxMLValueIdx() == ort_value_idx_map.MaxIdx(),
              "node_index_info and ort_value_idx_map are out of sync and cannot be used");
}

IExecutionFrame::~IExecutionFrame() = default;

#ifdef ENABLE_ATEN
Status IExecutionFrame::SetOutputMLValue(int index, const OrtValue& ort_value) {
  int ort_value_idx = GetNodeIdxToMLValueIdx(index);
  if (ort_value_idx == NodeIndexInfo::kInvalidEntry || static_cast<size_t>(ort_value_idx) >= all_values_size_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "invalid index ", ort_value_idx);
  }

  if (all_values_[ort_value_idx].IsAllocated()) {
    ORT_RETURN_IF_ERROR(CopyTensor(ort_value.Get<Tensor>(), *all_values_[ort_value_idx].GetMutable<Tensor>()));
  } else {
    all_values_[ort_value_idx] = ort_value;
  }
  return Status::OK();
}
#endif

#ifdef ENABLE_TRAINING
void IExecutionFrame::UpdateFeeds(gsl::span<const int> feed_mlvalue_idxs, gsl::span<const OrtValue> feeds) {
  ORT_ENFORCE(feed_mlvalue_idxs.size() == feeds.size());

  for (size_t idx = 0, end = feed_mlvalue_idxs.size(); idx < end; ++idx) {
    int ort_value_idx = feed_mlvalue_idxs[idx];
    // we are sharing the underlying tensor/object for OrtValue

    ORT_ENFORCE(!all_values_[ort_value_idx].IsAllocated());

    all_values_[ort_value_idx] = feeds[idx];
  }
}

void IExecutionFrame::UpdateFetches(gsl::span<const int> fetch_mlvalue_idxs,
                                    gsl::span<const OrtValue> fetches, const std::unordered_map<int, OrtValue>& initializers) {
  ORT_ENFORCE(fetch_mlvalue_idxs.size() == fetches.size());

  if (!fetches.empty()) {
    fetch_mlvalue_idxs_.assign(fetch_mlvalue_idxs.begin(), fetch_mlvalue_idxs.end());

    auto num_fetches = fetch_mlvalue_idxs_.size();

    for (size_t idx = 0; idx < num_fetches; ++idx) {
      int ort_value_idx = fetch_mlvalue_idxs_[idx];

      ORT_ENFORCE(!all_values_[ort_value_idx].IsAllocated());

      all_values_[ort_value_idx] = fetches[idx];

      // Copy the initializer if it is a fetch entry.
      auto entry = initializers.find(ort_value_idx);
      if (entry != initializers.end()) {
        const Tensor& src = entry->second.Get<Tensor>();
        OrtValue& dest = all_values_[ort_value_idx];

        if (!dest.IsAllocated()) {
          AllocatorPtr allocator = GetAllocator(src.Location().device);
          auto p_tensor = std::make_unique<Tensor>(src.DataType(), src.Shape(), allocator);
          auto ml_tensor = DataTypeImpl::GetType<Tensor>();
          dest.Init(p_tensor.release(), ml_tensor, ml_tensor->GetDeleteFunc());
        }

        ORT_THROW_IF_ERROR(CopyTensor(src, *dest.GetMutable<Tensor>()));
      }
    }
  }
}

Status IExecutionFrame::GetOutputs(gsl::span<const int> fetch_mlvalue_idxs, std::vector<OrtValue>& fetches) {
  auto num_fetches = fetch_mlvalue_idxs.size();

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
    fetches[idx] = GetMLValue(fetch_mlvalue_idxs[idx]);
  }

  return Status::OK();
}

#endif

// Return nullptr if index map to a value that is an unused optional input/output
const OrtValue* IExecutionFrame::GetNodeInputOrOutputMLValue(int index) const {
  int ort_value_idx = GetNodeIdxToMLValueIdx(index);
  return ort_value_idx != NodeIndexInfo::kInvalidEntry ? &(all_values_[ort_value_idx]) : nullptr;
}

OrtValue* IExecutionFrame::GetMutableNodeInputOrOutputMLValue(int index) {
  return const_cast<OrtValue*>(GetNodeInputOrOutputMLValue(index));
}

// TO DO: make it thread-safe
// This method is not thread-safe!
// Return S_OK and nullptr if index map to a value that is an unused optional input/output

Status IExecutionFrame::GetOrCreateNodeOutputMLValue(const int output_index, int output_arg_index,
                                                     const TensorShape* shape, OrtValue*& p_ort_value,
                                                     const Node& node) {
  auto status = Status::OK();
  int ort_value_idx = GetNodeIdxToMLValueIdx(output_arg_index);

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
      } else if (p_ort_value->IsSparseTensor()) {
#if !defined(DISABLE_SPARSE_TENSORS)
        const SparseTensor& sp_tensor = p_ort_value->Get<SparseTensor>();
        ORT_ENFORCE(shape && sp_tensor.DenseShape() == *shape,
                    "OrtValue shape verification failed. Current shape:", sp_tensor.DenseShape(),
                    " Requested shape:", shape ? shape->ToString() : "null");
#endif
      }
    } else {
      // shape is nullptr for traditional ML output values
      if (shape != nullptr && IsOutput(ort_value_idx)) {
        VerifyOutputSizes(output_index, node, *shape);
      }
      status = CreateNodeOutputMLValueImpl(*p_ort_value, ort_value_idx, shape);
    }
  }

  return status;
}

bool IExecutionFrame::TryGetInferredShape(int /*index*/, TensorShape& /*shape*/) const {
  // By default, there is no information about inferred shape, so this default
  // implementation always returns false. The derived class of IExecutionFrame
  // can override this function to provide, for example, activations' shape information.
  return false;
}

AllocatorPtr IExecutionFrame::GetAllocator(const OrtDevice& info) const {
  return GetAllocatorImpl(info);
}

Status IExecutionFrame::ReleaseMLValue(int ort_value_idx) { return ReleaseMLValueImpl(ort_value_idx); }

Status IExecutionFrame::ReleaseMLValueImpl(int ort_value_idx) {
  if (ort_value_idx == NodeIndexInfo::kInvalidEntry || static_cast<size_t>(ort_value_idx) >= all_values_size_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "invalid index ", ort_value_idx);
  }

  all_values_[ort_value_idx] = OrtValue();
  return Status::OK();
}

int IExecutionFrame::GetNodeIdxToMLValueIdx(int index) const {
  // The validity of the index is checked by GetMLValueIndex
  int ort_value_idx = node_index_info_.GetMLValueIndex(index);
  return ort_value_idx;
}

void IExecutionFrame::Init(gsl::span<const int> feed_mlvalue_idxs, gsl::span<const OrtValue> feeds,
                           const std::unordered_map<int, OrtValue>& initializers,
                           const std::function<bool(const std::string& name)>& is_initializer_sparse_func,
                           gsl::span<const OrtValue> fetches) {
  ORT_ENFORCE(feeds.size() == feed_mlvalue_idxs.size());
  ORT_ENFORCE(fetches.empty() || fetches.size() == fetch_mlvalue_idxs_.size());

  // Need this for sparse conversions in host memory
  AllocatorPtr cpu_allocator = GetAllocator(OrtDevice());

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

  // 3. Handle the weights.
  // We do this after the fetches to handle an edge case where an initializer is an output.
  // e.g. A Constant node gets lifted to an initializer so there's no Node producing the value as an output during
  // Graph execution (i.e. Graph execution won't write the value to all_values_).
  // A non-empty fetches vector will overwrite the actual weight in all_values_[ort_value_idx] if we did this earlier.
  // This makes the ONNX Constant test (onnx\backend\test\data\node\test_constant) happy as that
  // involves a graph with a single Constant node.
  for (const auto& entry : initializers) {
    int ort_value_index = entry.first;

    // If the initializer is an output we need to allocate or use a provided fetch buffer and copy the data
    //  so it can be returned to the caller.
    //
    //  The alternative to handling this as a special case would be to disallow an initializer providing a graph output.
    //  There's nothing in the ONNX spec that says a graph output must come from a node output though.
    //  If we took that approach we'd need to:
    //    - reject a model with an initializer or Constant node (as we convert those to initializers in Graph::Graph)
    //      that produces a graph output even though it conforms to the ONNX spec
    //    - update optimizers to not convert something to an initializer that is a graph output
    //      (e.g. constant folding)
    if (IsOutput(ort_value_index)) {
      std::string name;
      ORT_THROW_IF_ERROR(ort_value_idx_map_.GetName(ort_value_index, name));
      const Tensor& src = entry.second.Get<Tensor>();  // all initializers in ONNX are tensors
      OrtValue& dest = all_values_[ort_value_index];

#if !defined(DISABLE_SPARSE_TENSORS)
      const bool is_sparse_initializer = is_initializer_sparse_func(name);
      if (is_sparse_initializer) {
        if (!dest.IsAllocated()) {
          auto p_tensor = std::make_unique<SparseTensor>();
          auto ml_tensor = DataTypeImpl::GetType<SparseTensor>();
          dest.Init(p_tensor.release(), ml_tensor, ml_tensor->GetDeleteFunc());
        }

        // Outputting Coo format because initializers are Constant nodes, and they are converted to dense.
        AllocatorPtr allocator = GetAllocator(src.Location().device);
        constexpr bool has_linear_coo_index = true;
        ORT_THROW_IF_ERROR(sparse_utils::DenseTensorToSparseCoo(GetDataTransferManager(), src,
                                                                cpu_allocator, allocator, has_linear_coo_index,
                                                                *dest.GetMutable<SparseTensor>()));
      } else {
#else
      ORT_UNUSED_PARAMETER(is_initializer_sparse_func);
#endif  //  !defined(DISABLE_SPARSE_TENSORS)
        if (!dest.IsAllocated()) {
          // NOTE: This doesn't need to support ExecutionFrame custom allocators as they only come into play
          // for a subgraph with an output of unknown shape that needs to be accumulated by the control-flow node.
          // If the initializer is providing the output, the shape is known.
          AllocatorPtr allocator = GetAllocator(src.Location().device);
          Tensor::InitOrtValue(src.DataType(), src.Shape(), std::move(allocator), dest);
        }
        ORT_THROW_IF_ERROR(CopyTensor(src, *dest.GetMutable<Tensor>()));
#if !defined(DISABLE_SPARSE_TENSORS)
      }
#endif
    } else {
      all_values_[ort_value_index] = entry.second;
    }
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

ExecutionFrame::ExecutionFrame(gsl::span<const int> feed_mlvalue_idxs, gsl::span<const OrtValue> feeds,
                               gsl::span<const int> fetch_mlvalue_idxs, gsl::span<const OrtValue> fetches,
                               const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
#ifdef ORT_ENABLE_STREAM
                               const DeviceStreamCollection* device_streams,
#endif
                               const SessionState& session_state)
    : IExecutionFrame(session_state.GetOrtValueNameIdxMap(), session_state.GetNodeIndexInfo(), fetch_mlvalue_idxs),
#ifdef ORT_ENABLE_STREAM
      device_streams_(device_streams),
#endif
      session_state_(session_state),
      mem_patterns_(nullptr) {
  Init(
      feed_mlvalue_idxs, feeds, session_state.GetInitializedTensors(),
#if !defined(DISABLE_SPARSE_TENSORS)
      [&session_state](const std::string& name) -> bool {
        int idx = -1;
        if (session_state.GetOrtValueNameIdxMap().GetIdx(name, idx).IsOK()) {
          return session_state.IsSparseInitializer(idx);
        }
        return false;
      },
#else
      [&](const std::string& /*name*/) -> bool {
        return false;
      },
#endif
      fetches);

#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
  session_state.GetMemoryProfiler()->GetMemoryInfo().IncreaseIteration();
#endif

  // map the custom allocators to ort_value_idx entries
  if (!fetch_allocators.empty()) {
    custom_allocators_.reserve(fetch_allocators.size());
    const auto idx_size = fetch_mlvalue_idxs.size();
    for (const auto& e : fetch_allocators) {
      if (e.first < idx_size) {
        int ort_value_idx = fetch_mlvalue_idxs[e.first];
        custom_allocators_.insert_or_assign(ort_value_idx, e.second);
      }
    }
  }

  // If the session enable memory pattern optimization
  // and we have execution plan generated, try to setup
  // memory pattern optimization.
  if (session_state.GetEnableMemoryPattern() && session_state.GetExecutionPlan()) {
    bool all_tensors = true;
    // Reserve mem to avoid re-allocation.
    for (const auto& feed : feeds) {
      if (!feed.IsTensor()) {
        all_tensors = false;
        break;
      }
    }

    // if there are some traditional ml value type in inputs disable the memory pattern optimization.
    if (all_tensors) {
      mem_patterns_ = session_state.GetMemoryPatternGroup(feeds, feed_mlvalue_idxs, inferred_shapes_);
      // if no existing patterns, generate one in this execution frame
      if (!mem_patterns_) {
        planner_.emplace(*session_state.GetExecutionPlan());
      } else {
        // pre-allocate the big chunk requested in memory pattern.
        // all the internal kernel's input/output tensors will be allocated on these buffer.
        buffers_.reserve(mem_patterns_->locations.size());
        for (size_t i = 0; i < mem_patterns_->locations.size(); i++) {
          const auto& location = mem_patterns_->locations[i];
          ORT_ENFORCE(buffers_.find(location) == buffers_.end());
          if (mem_patterns_->patterns[i].PeakSize() > 0) {
            AllocatorPtr alloc = GetAllocator(location);
            void* buffer = nullptr;
            // it's possible we can't allocate the large block. if we have memory patterns we know we have successfully
            // executed once before, so if there's an arena involved it probably has smaller blocks available.
            // due to that we can still run and use those blocks (inside the arena logic) instead of one large one.
            // it's less efficient (the arena will add some overhead to coalesce individual allocations
            // back into blocks on 'free'), but better than failing completely.
            ORT_TRY {
              auto peak_size = mem_patterns_->patterns[i].PeakSize();
              // Planning of one memory type should only happen once.
#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
              ORT_ENFORCE(
                  static_activation_memory_sizes_in_byte_.find(location.ToString()) ==
                      static_activation_memory_sizes_in_byte_.end(),
                  "Memory type ",
                  location.ToString(),
                  " should only appear once.");
              // static_activation_memory_in_bytes_ is max virtual memory size the planner computes.
              // Memory dynamically allocated when executing kernels is not recorded using this field.
              static_activation_memory_sizes_in_byte_[location.ToString()] = peak_size;
#endif
              // the memory pattern buffer will leave in the whole execution.
#ifdef ORT_ENABLE_STREAM
              StreamAwareArena* stream_aware_alloc = AsStreamBasedAllocator(alloc);
              if (stream_aware_alloc && device_streams_) {
                Stream* mem_pattern_stream = device_streams_->GetRootStream();
                buffer = stream_aware_alloc->AllocOnStream(peak_size, mem_pattern_stream, nullptr);
                for (size_t j = 0; j < device_streams_->NumStreams(); j++) {
                  stream_aware_alloc->SecureTheChunk(mem_pattern_stream, device_streams_->GetStream(j), nullptr);
                }
              } else {
                buffer = alloc->Alloc(peak_size);
              }
#else
              buffer = alloc->Alloc(peak_size);
#endif
              // handle allocator that doesn't throw
              if (buffer == nullptr) {
                // INFO level as this may fire on every run and there may not be much a user can do
                LOGS(session_state_.Logger(), INFO) << "Allocation of memory pattern buffer for "
                                                    << location.ToString() << " returned nullptr";
              }
            }
            ORT_CATCH(const OnnxRuntimeException& ex) {
              ORT_HANDLE_EXCEPTION([&]() {
                LOGS(session_state_.Logger(), INFO) << "Allocation of memory pattern buffer for "
                                                    << location.ToString() << " failed. Error:" << ex.what();
              });
            }

            if (buffer != nullptr) {
              buffers_[location] = BufferUniquePtr(buffer, BufferDeleter(alloc));
            }
#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
            // Record activation memory pattern
            auto mem_profier_ptr = session_state.GetMemoryProfiler();
            mem_profier_ptr->GetMemoryInfo().ClearMemoryInfoPerExecution();
            if (mem_patterns_ && buffer != nullptr) {
              mem_profier_ptr->GetMemoryInfo().RecordPatternInfo(*mem_patterns_, MemoryInfo::MapType::StaticActivation);
              mem_profier_ptr->CreateEvents(
                  "static activations_" + std::to_string(mem_profier_ptr->GetMemoryInfo().GetIteration()),
                  mem_profier_ptr->GetAndIncreasePid(), MemoryInfo::MapType::StaticActivation, "", 0);
            }
#endif
            // log size of activation. Keep it commented out for now to avoid log flooding.
            // VLOGS(session_state_.Logger(), 1) << "**** Allocated memory for activations, size: "
            //                                   << mem_patterns_->patterns[i].PeakSize();
          }
        }
      }
    }
  }
}

ExecutionFrame::~ExecutionFrame() = default;

Status ExecutionFrame::CopyTensor(const Tensor& src, Tensor& dest) const {
  return session_state_.GetDataTransferMgr().CopyTensor(src, dest);
}

const DataTransferManager& ExecutionFrame::GetDataTransferManager() const {
  return session_state_.GetDataTransferMgr();
}

Status ExecutionFrame::AllocateMLValueTensorSelfOwnBuffer(OrtValue& ort_value, int ort_value_index,
                                                          MLDataType element_type, const OrtDevice& location,
                                                          const TensorShape& shape) {
  return AllocateMLValueTensorSelfOwnBufferHelper(ort_value, ort_value_index, element_type, location, shape);
}

Stream* ExecutionFrame::GetValueStream(int ort_value_idx) const {
#ifdef ORT_ENABLE_STREAM
  const auto& value_to_stream_map = const_cast<SessionState&>(session_state_).GetExecutionPlan()->GetValueToStreamMap();
  auto it = value_to_stream_map.find(ort_value_idx);
  if (it != value_to_stream_map.end() && device_streams_ != nullptr && it->second < device_streams_->NumStreams()) {
    return device_streams_->GetStream(it->second);
  }
#else
  ORT_UNUSED_PARAMETER(ort_value_idx);
#endif
  return nullptr;
}

Status ExecutionFrame::AllocateMLValueTensorSelfOwnBufferHelper(OrtValue& ort_value, int ort_value_index,
                                                                MLDataType element_type,
                                                                const OrtDevice& location,
                                                                const TensorShape& shape) {
  if (ort_value_index == NodeIndexInfo::kInvalidEntry) {
    return Status(ONNXRUNTIME, FAIL, "Trying to allocate memory for unused optional inputs/outputs");
  }

  size_t size;
  int64_t len = shape.Size();
  if (len < 0) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Tensor shape cannot contain any negative value");
  }
  if (static_cast<uint64_t>(len) > std::numeric_limits<size_t>::max()) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Tensor shape is too large");
  }
  if (!IAllocator::CalcMemSizeForArrayWithAlignment<kAllocAlignment>(static_cast<size_t>(len), element_type->Size(), &size)) {
    return Status(ONNXRUNTIME, FAIL, "size overflow");
  }

  // Lazily get the allocator only if needed.
  AllocatorPtr alloc = nullptr;

  // if we have pre-calculated memory pattern, and the ort_value is not output mlvalue
  // try to allocate on pre-allocated big chunk.
  const auto& per_alloc_plan = GetAllocationPlan(ort_value_index);

  if (mem_patterns_ && per_alloc_plan.alloc_kind != AllocKind::kAllocateOutput &&
      per_alloc_plan.alloc_kind != AllocKind::kAllocatedExternally) {
    auto pattern = mem_patterns_->GetPatterns(location);
    if (pattern) {
      auto block = pattern->GetBlock(ort_value_index);
      // if block not found, fall back to default behavior
      if (block) {
        auto it = buffers_.find(location);
        if (it != buffers_.end()) {
          // if the block is not correct, log message then fall back to default behavior
          if (block->size_ == size) {
            void* buffer = it->second.get();
            auto status = AllocateTensorWithPreAllocateBufferHelper(
                ort_value, static_cast<void*>(static_cast<char*>(buffer) + block->offset_), element_type, location,
                shape);
            return status;
          } else {
            // the block size may vary especially if the model has NonZero ops, or different sequence lengths are
            // fed in, so use VERBOSE as the log level as it's expected.
            // TODO: Should we reuse the block if the size is large enough? Would probably need to allow it
            // to be freed if the size difference was too large so our memory usage doesn't stick at a high water mark
            LOGS(session_state_.Logger(), VERBOSE) << "For ort_value with index: " << ort_value_index
                                                   << ", block in memory pattern size is: " << block->size_
                                                   << " but the actual size is: " << size
                                                   << ", fall back to default allocation behavior";
          }
        }
        // else { we couldn't allocate the large block for the buffer so we didn't insert an entry }
      }
    }
  }

  // no memory pattern, or the pattern is not correct.
  if (!alloc) alloc = GetAllocator(location);
  ORT_ENFORCE(alloc && alloc.get() != nullptr, "Failed to get allocator for ", location.ToString());

  Stream* current_stream = GetValueStream(ort_value_index);
  if (current_stream) {
#ifdef ORT_ENABLE_STREAM
    auto stream_aware_alloc = AsStreamBasedAllocator(alloc);
    if (stream_aware_alloc) {
      size_t buffer_size = Tensor::CalculateTensorStorageSize(element_type, shape);
      // the reused memory must from same EP
      auto wait_handle = this->session_state_.GetStreamHandleRegistryInstance().GetWaitHandle(
          current_stream->GetDevice().Type(), current_stream->GetDevice().Type());
      void* p_data = stream_aware_alloc->AllocOnStream(buffer_size, current_stream, wait_handle);
      Tensor::InitOrtValue(element_type, shape, p_data, std::move(alloc), ort_value);
    } else {
      Tensor::InitOrtValue(element_type, shape, std::move(alloc), ort_value);
    }
#else
    ORT_THROW("Ort value is associated with a Stream but Stream is not enabled in the build.");
#endif
  } else {
    Tensor::InitOrtValue(element_type, shape, std::move(alloc), ort_value);
  }

  // trace the memory allocation.
  // don't trace the memory allocation on string tensors, as it need
  // placement new, we don't support it in memory pattern optimization.
  if (!utils::IsDataTypeString(element_type)) {
    TraceAllocate(ort_value_index, size);
  }

  {
#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
    // This code block is not thread-safe.
    // Dynamic activation size would be accessed by multiple threads
    // if parallel executor is used.
    std::unique_lock<std::mutex> lock(mtx_);
    dynamic_activation_memory_sizes_in_byte_[location.ToString()] += size;
    session_state_.GetMemoryProfiler()->GetMemoryInfo().SetDynamicAllocation(ort_value_index);
#endif
  }

  return Status::OK();
}

Status ExecutionFrame::AllocateMLValueTensorPreAllocateBuffer(OrtValue& ort_value, int ort_value_index_reuse,
                                                              MLDataType element_type, const OrtDevice& location,
                                                              const TensorShape& shape,
                                                              bool is_strided_tensor) {
  OrtValue& ort_value_reuse = GetMutableMLValue(ort_value_index_reuse);

  auto* reuse_tensor = ort_value_reuse.GetMutable<Tensor>();

  // Training starts to support strided tensor that the shape size may be larger (like Expand), smaller (like Split) or
  // equal (like Transpose) to the shared tensor's shape size, so below check is no longer valid.
#ifndef ENABLE_STRIDED_TENSORS
  ORT_ENFORCE(!is_strided_tensor);
#endif  // ENABLE_STRIDED_TENSORS
  if (!is_strided_tensor) {
    auto buffer_num_elements = reuse_tensor->Shape().Size();
    auto required_num_elements = shape.Size();

    // check number of elements matches. shape may not be an exact match (e.g. Reshape op)
    if (buffer_num_elements != required_num_elements) {
      // could be an allocation planner bug (less likely) or the model incorrectly uses something like 'None'
      // as a dim_param, or -1 in dim_value in multiple places making the planner think those shapes are equal.
      auto message = onnxruntime::MakeString(
          "Shape mismatch attempting to re-use buffer. ", reuse_tensor->Shape(), " != ", shape,
          ". Validate usage of dim_value (values should be > 0) and "
          "dim_param (all values with the same string should equate to the same size) in shapes in the model.");

      // be generous and use the buffer if it's large enough. log a warning though as it indicates a bad model
      if (buffer_num_elements >= required_num_elements) {
        // View Operator is reusing the buffer bigger than the required size.
        // Disabling warning message for now. The op is in the process of being deprecated.
#ifndef ENABLE_TRAINING
        LOGS(session_state_.Logger(), WARNING) << message;
#endif  // ENABLE_TRAINING
      } else {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, message);
      }
    }
  }

  void* reuse_buffer = reuse_tensor->MutableDataRaw();

  return AllocateTensorWithPreAllocateBufferHelper(ort_value, reuse_buffer, element_type, location, shape);
}

Status ExecutionFrame::AllocateTensorWithPreAllocateBufferHelper(OrtValue& ort_value, void* pBuffer,
                                                                 MLDataType element_type,
                                                                 const OrtDevice& location,
                                                                 const TensorShape& shape) {
  Tensor::InitOrtValue(element_type, shape, pBuffer, GetAllocator(location)->Info(), ort_value, 0L);
  return Status::OK();
}

static Status AllocateTraditionalMLValue(OrtValue& ort_value, const NonTensorTypeBase& type) {
  auto creator = type.GetCreateFunc();
  ort_value.Init(creator(), &type, type.GetDeleteFunc());
  return Status::OK();
}

static Status AllocateTensorSequence(OrtValue& ort_value) {
  auto ml_tensor_sequence = DataTypeImpl::GetType<TensorSeq>();
  auto p_tensor_sequence = std::make_unique<TensorSeq>();
  ort_value.Init(p_tensor_sequence.release(), ml_tensor_sequence, ml_tensor_sequence->GetDeleteFunc());

  return Status::OK();
}

#if !defined(DISABLE_SPARSE_TENSORS)
static Status AllocateSparseTensor(OrtValue& mlvalue, const DataTypeImpl& ml_type, AllocatorPtr allocator,
                                   const TensorShape& shape,
                                   const SessionState& /*session_state*/) {
  auto element_type = ml_type.AsSparseTensorType()->GetElementType();
  SparseTensor::InitOrtValue(element_type, shape, std::move(allocator), mlvalue);

  return Status::OK();
}
#endif

Status ExecutionFrame::AllocateReusedOrtValueIfNotAllocatedHelper(int reuse_mlvalue_index, const TensorShape* shape) {
  // In case OrtRunOptions.only_execute_path_to_fetches == true, it is possible that 'reuse_value'
  // is not allocated (its upstream op is not executed due to the option).
  // In this case we need to allocate 'reuse_value' and then let 'ort_value' to reuse it.
  OrtValue& reuse_value = GetMutableMLValue(reuse_mlvalue_index);
  if (!reuse_value.IsAllocated()) {
    ORT_RETURN_IF_ERROR(AllocateAsPerAllocationPlan(reuse_value, reuse_mlvalue_index, shape));
  }

  return Status::OK();
}

// This method is not thread safe!
Status ExecutionFrame::AllocateAsPerAllocationPlan(OrtValue& ort_value, int ort_value_index, const TensorShape* shape) {
  const auto& alloc_plan = session_state_.GetPerValueAllocPlan();
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

  if (ml_type->IsTensorType()
#if !defined(DISABLE_OPTIONAL_TYPE)
      || utils::IsOptionalTensor(ml_type)
#endif
  ) {
    ORT_ENFORCE(shape, "Allocation of tensor types requires a shape.");

    // tensors / optional tensors
#if !defined(DISABLE_OPTIONAL_TYPE)
    const auto* ml_data_type = ml_type->IsTensorType()
                                   ? static_cast<const TensorTypeBase*>(ml_type)->GetElementType()
                                   : utils::GetElementTypeFromOptionalTensor(ml_type);
#else
    const auto* ml_data_type = static_cast<const TensorTypeBase*>(ml_type)->GetElementType();
#endif

    AllocKind alloc_kind = per_alloc_plan.alloc_kind;
    switch (alloc_kind) {
      // Right now for kAllocate and kAllocateOutput we are using same approach.
      // In the future we may want to have different way to handle it.
      case AllocKind::kAllocateOutput:
      case AllocKind::kAllocate: {
        ORT_RETURN_IF_ERROR(AllocateMLValueTensorSelfOwnBuffer(ort_value, ort_value_index, ml_data_type, alloc_info,
                                                               *shape));
        break;
      }
      case AllocKind::kReuse: {
        int reuse_mlvalue_index = per_alloc_plan.reused_buffer;

        ORT_RETURN_IF_ERROR(AllocateReusedOrtValueIfNotAllocatedHelper(reuse_mlvalue_index, shape));

        bool is_strided_tensor = false;
#ifdef ENABLE_STRIDED_TENSORS
        is_strided_tensor = per_alloc_plan.is_strided_tensor;
#endif  // ENABLE_STRIDED_TENSORS
        ORT_RETURN_IF_ERROR(AllocateMLValueTensorPreAllocateBuffer(
            ort_value, reuse_mlvalue_index, ml_data_type, alloc_info, *shape, is_strided_tensor));
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

#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
    session_state_.GetMemoryProfiler()->GetMemoryInfo().RecordActivationAllocInfo(ort_value_index, ort_value);
#endif

    return Status::OK();
  } else if (ml_type->IsSparseTensorType()) {
#if !defined(DISABLE_SPARSE_TENSORS)
    return AllocateSparseTensor(ort_value, *ml_type, GetAllocator(alloc_info),
                                *shape, session_state_);
#else
    // Model load should have failed so this should be unreachable
    ORT_THROW("SparseTensor is not supported in this build.");
#endif
  } else if (ml_type->IsTensorSequenceType()
#if !defined(DISABLE_OPTIONAL_TYPE)
             || utils::IsOptionalSeqTensor(ml_type)
#endif
  ) {
    AllocKind alloc_kind = per_alloc_plan.alloc_kind;

    if (alloc_kind == AllocKind::kReuse) {
      int reuse_mlvalue_index = per_alloc_plan.reused_buffer;

      ORT_RETURN_IF_ERROR(AllocateReusedOrtValueIfNotAllocatedHelper(reuse_mlvalue_index, shape));

      OrtValue& reuse_value = GetMutableMLValue(reuse_mlvalue_index);

      // copy at the OrtValue level so the shared_ptr for the data is shared between the two OrtValue instances
      ort_value = reuse_value;

      return Status::OK();
    } else {
      return AllocateTensorSequence(ort_value);
    }
  } else {
    return AllocateTraditionalMLValue(ort_value, *static_cast<const NonTensorTypeBase*>(ml_type));
  }
}

AllocatorPtr ExecutionFrame::GetAllocatorImpl(const OrtDevice& info) const {
  return session_state_.GetAllocator(info);
}

// This method is not thread safe!
// Return S_OK and nullptr if index map to a value that is an unused optional input/output
Status ExecutionFrame::CreateNodeOutputMLValueImpl(OrtValue& ort_value, int ort_value_idx, const TensorShape* shape) {
  return AllocateAsPerAllocationPlan(ort_value, ort_value_idx, shape);
}

void ExecutionFrame::VerifyOutputSizes(int output_index, const Node& node, const TensorShape& output_shape) {
  const NodeArg* output_def = node.OutputDefs()[output_index];
  const auto* expected_shape = output_def->Shape();
  if (expected_shape == nullptr) {
    // model didn't specify shape and shape inferencing wasn't able to calculate it so nothing to compare against
    return;
  }

  const size_t expected_rank = expected_shape->dim_size();
  bool compatible = expected_rank == output_shape.NumDimensions();
  if (compatible) {
    for (size_t i = 0; i < expected_rank; ++i) {
      const auto& expected_dim = expected_shape->dim().Get(static_cast<int>(i));
      if (expected_dim.has_dim_value() && expected_dim.dim_value() != output_shape[i]) {
        compatible = false;
        break;
      }
    }
  }

  if (!compatible) {
    LOGS(session_state_.Logger(), WARNING)
        << "Expected shape from model of " << utils::GetTensorShapeFromTensorShapeProto(*expected_shape)
        << " does not match actual shape of " << output_shape << " for output " << output_def->Name();
  }
}

// do not call this in ParallExecutionPlan
Status ExecutionFrame::ReleaseMLValueImpl(int ort_value_idx) {
  ORT_RETURN_IF_ERROR(IExecutionFrame::ReleaseMLValueImpl(ort_value_idx));
  TraceFree(ort_value_idx);
  return Status::OK();
}

const AllocPlanPerValue& ExecutionFrame::GetAllocationPlan(int ort_value_idx) {
  return session_state_.GetPerValueAllocPlan()[ort_value_idx];
}

void ExecutionFrame::TraceAllocate(int ort_value_idx, size_t size) {
  if (planner_.has_value()) {
    // don't trace the output tensors or external outputs.
    auto& allocation_plan = GetAllocationPlan(ort_value_idx);
    if (allocation_plan.alloc_kind == AllocKind::kAllocateOutput ||
        allocation_plan.alloc_kind == AllocKind::kAllocatedExternally) {
      return;
    }
    auto status = planner_->TraceAllocation(ort_value_idx, size);
    if (!status.IsOK()) {
      LOGS(session_state_.Logger(), WARNING) << "TraceAllocation for ort_value_idx=" << ort_value_idx
                                             << " size=" << size << " failed: " << status.ErrorMessage();
    }
  }
}

// do not call this in ParallExecutionPlan
void ExecutionFrame::TraceFree(int ort_value_idx) {
  // don't trace free on output tensors.
  if (planner_.has_value() && !IsOutput(ort_value_idx)) {
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
Status ExecutionFrame::GeneratePatterns(MemoryPatternGroup& out) {
  if (!planner_.has_value()) {
    return Status(ONNXRUNTIME, FAIL, "Memory pattern planner is not enabled on this execution framework.");
  }

  return planner_->GeneratePatterns(out);
}

bool ExecutionFrame::TryGetInferredShape(int index, TensorShape& shape) const {
  // NodeArg index to OrtValue index.
  int ort_value_idx = GetNodeIdxToMLValueIdx(index);

  // Check if index is valid.
  if (ort_value_idx == NodeIndexInfo::kInvalidEntry) {
    return false;
  }

  // Search for inferred shape.
  // If the inferred shape is found, it's assigned to "shape" so that caller can use it.
  if (inferred_shapes_ != nullptr) {
    auto it = inferred_shapes_->find(ort_value_idx);
    if (it != inferred_shapes_->end()) {
      shape = it->second;
      return true;
    }
  }

  // Tell the caller if the search is successful or not.
  return false;
}

}  // namespace onnxruntime
