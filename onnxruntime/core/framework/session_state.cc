// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/session_state.h"

#include <sstream>

#include "core/platform/ort_mutex.h"
#include "core/common/logging/logging.h"
#include "core/common/safeint.h"
#include "core/flatbuffers/schema/ort.fbs.h"
#include "core/framework/allocator.h"
#include "core/framework/kernel_def_hash_helpers.h"
#include "core/framework/node_index_info.h"
#include "core/framework/op_kernel.h"
#include "core/framework/ort_value_pattern_planner.h"
#include "core/framework/session_state_flatbuffers_utils.h"
#include "core/framework/session_state_utils.h"
#include "core/framework/utils.h"
#include "core/providers/cpu/controlflow/utils.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

using namespace ::onnxruntime::common;

namespace onnxruntime {

void SessionState::SetupAllocators() {
  // register allocators in reverse order. one per OrtMemType+OrtDevice combination.
  // this order prefers allocators from the core EPs (CPU EP, CUDA EP, ROCM EP) and ensures the CUDA/ROCM EP's
  // per-thread allocators will be preferred over the TensorRT/MIGraphX EP non-per-thread allocators.
  // TODO: Refactor the EP/allocator relationship so that we can be explicit about which allocator is preferred and
  //       avoid creating unnecessary allocators.
  std::for_each(std::make_reverse_iterator(execution_providers_.end()),
                std::make_reverse_iterator(execution_providers_.begin()),
                [this](const auto& provider_iter) {
                  IExecutionProvider& provider = *provider_iter;
                  for (const auto& allocator : provider.GetAllocators()) {
                    const OrtMemoryInfo& memory_info = allocator->Info();
                    auto iter = allocators_.find(memory_info);
                    if (iter != allocators_.end()) {
                      // EPs could be sharing allocators so no info message unless this is a different instance.
                      // This is an expected scenario as multiple EPs may have allocators for the same device.
                      // e.g. xnnpack and CPU EP are both CPU based.
                      if (iter->second(memory_info.id, memory_info.mem_type) != allocator) {
                        LOGS(logger_, INFO) << "Allocator already registered for " << allocator->Info()
                                            << ". Ignoring allocator from " << provider.Type();
                      }
                    } else {
                      // slightly weird indirection to go back to the provider to get the allocator each time
                      // it's needed in order to support scenarios such as the CUDA EP's per-thread allocator.
                      allocators_[memory_info] = [&provider](int id, OrtMemType mem_type) {
                        return provider.GetAllocator(id, mem_type);
                      };
                    }
                  }
                });
}

AllocatorPtr SessionState::GetAllocator(const OrtMemoryInfo& location) const noexcept {
  AllocatorPtr result;
  auto entry = allocators_.find(location);
  if (entry != allocators_.cend()) {
    result = entry->second(location.id, location.mem_type);
  }

  return result;
}

AllocatorPtr SessionState::GetAllocator(OrtDevice device) const noexcept {
  for (const auto& iter : allocators_) {
    if (iter.first.device == device) {
      return iter.second(device.Id(), iter.first.mem_type);
    }
  }
  return nullptr;
}

void SessionState::CreateGraphInfo() {
  graph_viewer_.emplace(graph_);
  // use graph_viewer_ to initialize ort_value_name_idx_map_
  LOGS(logger_, VERBOSE) << "SaveMLValueNameIndexMapping";
  int idx = 0;

  // we keep all graph inputs (including initializers), even if they are unused, so make sure they all have an entry
  for (const auto* input_def : graph_viewer_->GetInputsIncludingInitializers()) {
    idx = ort_value_name_idx_map_.Add(input_def->Name());
    VLOGS(logger_, 1) << "Added graph_viewer_ input with name: " << input_def->Name()
                      << " to OrtValueIndex with index: " << idx;
  }

  for (auto& node : graph_viewer_->Nodes()) {
    // build the OrtValue->index map
    for (const auto* input_def : node.InputDefs()) {
      if (input_def->Exists()) {
        idx = ort_value_name_idx_map_.Add(input_def->Name());
        VLOGS(logger_, 1) << "Added input argument with name: " << input_def->Name()
                          << " to OrtValueIndex with index: " << idx;
      }
    }

    for (const auto* input_def : node.ImplicitInputDefs()) {
      if (input_def->Exists()) {
        idx = ort_value_name_idx_map_.Add(input_def->Name());
        VLOGS(logger_, 1) << "Added implicit input argument with name: " << input_def->Name()
                          << " to OrtValueIndex with index: " << idx;
      }
    }

    for (const auto* output_def : node.OutputDefs()) {
      if (output_def->Exists()) {
        ort_value_name_idx_map_.Add(output_def->Name());
        VLOGS(logger_, 1) << "Added output argument with name: " << output_def->Name()
                          << " to OrtValueIndex with index: " << idx;
      }
    }
  }

  // allocate OrtValue for graph outputs when coming from initializers
  for (const auto& output : graph_viewer_->GetOutputs()) {
    if (output->Exists()) {
      idx = ort_value_name_idx_map_.Add(output->Name());
      VLOGS(logger_, 1) << "Added graph output with name: " << output->Name()
                        << " to OrtValueIndex with index: " << idx;
    }
  }

  LOGS(logger_, VERBOSE) << "Done saving OrtValue mappings.";
}

#if !defined(ORT_MINIMAL_BUILD)
Status SessionState::PopulateKernelCreateInfo(const KernelRegistryManager& kernel_registry_manager,
                                              bool saving_ort_format) {
  for (auto& node : graph_.Nodes()) {
    const KernelCreateInfo* kci = nullptr;

    auto status = kernel_registry_manager.SearchKernelRegistry(node, &kci);
    if (!status.IsOK() && saving_ort_format) {
      // if we didn't find the kernel and are saving to ORT format an EP that compiles nodes is enabled.
      // in that case we assigned the node to that EP but do not compile it into a fused node.
      // this keeps the original node and prevents level 2 and level 3 optimizers from modifying it.
      // we now revert to the CPU EP to include the hash for the kernel as a fallback. at runtime when the model
      // is loaded in a minimal build, the compiling EP will replace this node if possible. if that's not possible for
      // some reason we can fallback to the CPU EP implementation via this hash.
      node.SetExecutionProviderType(kCpuExecutionProvider);
      status = kernel_registry_manager.SearchKernelRegistry(node, &kci);
    }

    ORT_RETURN_IF_ERROR(status);

    ORT_IGNORE_RETURN_VALUE(
        kernel_create_info_map_.insert({node.Index(), gsl::not_null<const KernelCreateInfo*>(kci)}));
  }

  for (const auto& entry : subgraph_session_states_) {
    for (const auto& name_to_subgraph_session_state : entry.second) {
      SessionState& subgraph_session_state = *name_to_subgraph_session_state.second;
      ORT_RETURN_IF_ERROR(subgraph_session_state.PopulateKernelCreateInfo(kernel_registry_manager, saving_ort_format));
    }
  }

  return Status::OK();
}
#endif

const KernelCreateInfo& SessionState::GetNodeKernelCreateInfo(NodeIndex node_index) const {
  auto entry = kernel_create_info_map_.find(node_index);
  // invalid node index or FinalizeSessionState should have been called. Either way it's an internal logic error
  ORT_ENFORCE(entry != kernel_create_info_map_.cend());

  return *entry->second;
}

Status SessionState::CreateKernels(const KernelRegistryManager& kernel_registry_manager) {
  const auto& nodes = graph_viewer_->Nodes();
  if (!nodes.empty()) {
    size_t max_nodeid = 0;
    for (const auto& node : nodes) {
      max_nodeid = std::max(max_nodeid, node.Index());
    }
    session_kernels_.clear();
    session_kernels_.resize(max_nodeid + 1);
    for (const auto& node : nodes) {
      // construct and save the kernels
      const KernelCreateInfo& kci = GetNodeKernelCreateInfo(node.Index());

      // the execution provider was required to be valid to find the KernelCreateInfo so we don't need to check it here
      onnxruntime::ProviderType exec_provider_name = node.GetExecutionProviderType();
      const IExecutionProvider& exec_provider = *execution_providers_.Get(exec_provider_name);

      // assumes vector is already resize()'ed to the number of nodes in the graph
      ORT_RETURN_IF_ERROR(kernel_registry_manager.CreateKernel(node, exec_provider, *this, kci, session_kernels_[node.Index()]));
    }
  }
  node_index_info_.emplace(*graph_viewer_, ort_value_name_idx_map_);
  return Status::OK();
}

const SequentialExecutionPlan* SessionState::GetExecutionPlan() const {
  if (!p_seq_exec_plan_.has_value()) {
    return nullptr;
  }
  return &p_seq_exec_plan_.value();
}

Status SessionState::AddInitializedTensor(int ort_value_index, const OrtValue& ort_value, const OrtCallback* d,
                                          bool constant, bool sparse) {
  auto p = initialized_tensors_.insert({ort_value_index, ort_value});
  if (!p.second)
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "duplicated ort_value index:", ort_value_index,
                           ". Do you have duplicated calls to SessionState::AddInitializedTensor function?");

  if (d != nullptr && d->f != nullptr) {
    deleter_for_initialized_tensors_.insert_or_assign(ort_value_index, *d);
  }

  if (constant) {
    constant_initialized_tensors_.insert({ort_value_index, ort_value});
  }

#if !defined(DISABLE_SPARSE_TENSORS)
  if (sparse) {
    sparse_initialized_tensors_.insert(ort_value_index);
  }
#else
  ORT_UNUSED_PARAMETER(sparse);
#endif

  return Status::OK();
}

const std::unordered_map<int, OrtValue>& SessionState::GetInitializedTensors() const { return initialized_tensors_; }

const std::unordered_map<int, OrtValue>& SessionState::GetConstantInitializedTensors() const {
  return constant_initialized_tensors_;
}

#if !defined(DISABLE_SPARSE_TENSORS)
bool SessionState::IsSparseInitializer(int ort_value_index) const {
  return sparse_initialized_tensors_.count(ort_value_index) > 0;
}
#endif

#ifdef ENABLE_TRAINING
Status SessionState::GetInitializedTensors(
    const std::unordered_set<std::string>& interested_weights,
    bool allow_missing_weights, NameMLValMap& retrieved_weights) const {
  NameMLValMap result;
  result.reserve(interested_weights.size());
  for (const auto& weight_name : interested_weights) {
    int idx;
    const auto status = GetOrtValueNameIdxMap().GetIdx(weight_name, idx);
    if (!status.IsOK()) {
      ORT_RETURN_IF_NOT(
          allow_missing_weights,
          "Failed to get OrtValue index from name: ", status.ErrorMessage());
      continue;
    }
    if (initialized_tensors_.find(idx) != initialized_tensors_.end()) {
      result.emplace(weight_name, initialized_tensors_.at(idx));
    } else {
      ORT_RETURN_IF_NOT(
          allow_missing_weights,
          "Failed to get initializer with name: ", weight_name, " and index:", idx);
      continue;
    }
  }
  retrieved_weights = std::move(result);
  return Status::OK();
}

NameMLValMap SessionState::GetInitializedTensors(const std::unordered_set<std::string>& interested_weights) const {
  NameMLValMap result;
  const auto status = GetInitializedTensors(interested_weights, true, result);
  ORT_ENFORCE(status.IsOK(), status.ErrorMessage());
  return result;
}
#endif

void SessionState::CleanInitializedTensorsFromGraph() {
  graph_.CleanAllInitializedTensors();
}

static Status KernelUseSharedPrePackedBuffers(OpKernel& kernel, int input_idx,
                                              const PrePackedWeights& prepacked_weights,
                                              const std::string& node_name) {
  std::vector<BufferUniquePtr> shared_prepacked_buffers;
  shared_prepacked_buffers.reserve(4);  // Unlikely to see more than 4 prepacked buffers per initializer

  for (const auto& prepacked_buffer : prepacked_weights.buffers_) {
    // BufferDeleter is nullptr because the kernel should not delete the shared buffer - it can only use it
    shared_prepacked_buffers.emplace_back(prepacked_buffer.get(), BufferDeleter(nullptr));
  }

  bool used_shared_buffers = false;
  ORT_RETURN_IF_ERROR(kernel.UseSharedPrePackedBuffers(shared_prepacked_buffers, input_idx, used_shared_buffers));

  // BUG CHECK: Ensure that the kernel used the provided shared buffers
  // Mostly a debug check to ensure that the kernel has an overridden implementation of the
  // base UseSharedPrePackedBuffers() which is basically a no-op.
  if (!used_shared_buffers)
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "The kernel corresponding to the node ", node_name,
                           " doesn't have an implementation that can consume provided pre-packed weights");

  return Status::OK();
}

static std::string GenerateKeyForPrepackedWeightsMap(const std::string& op_type,
                                                     const PrePackedWeights& pre_packed_weights) {
  std::ostringstream ss_1;
  ss_1 << op_type;
  ss_1 << "+";
  ss_1 << std::to_string(pre_packed_weights.GetHash());

  return ss_1.str();
}

Status SessionState::PrepackConstantInitializedTensors(InlinedHashMap<std::string, size_t>& constant_initializers_use_count,
                                                       const std::unordered_map<std::string, const OrtValue*>& initializers_to_share_map) {
  auto prepacked_constant_weights = [this, &constant_initializers_use_count, &initializers_to_share_map](
                                        bool should_cache_prepacked_weights_for_shared_initializers) -> Status {
    for (auto& node : GetGraphViewer().Nodes()) {
      auto kernel = GetMutableKernel(node.Index());
      int input_idx = 0;
      for (auto& input_def : node.InputDefs()) {
        if (input_def->Exists()) {
          const std::string& input_name = input_def->Name();
          SessionState* st = this;
          // subgraph can use the value from outer scope,
          // so it needs to check if current node uses constant initialized tensor from current and outer graphs
          do {
            int ort_value_idx;
            if (st->GetOrtValueNameIdxMap().GetIdx(input_name, ort_value_idx).IsOK()) {
              std::unordered_map<int, OrtValue>& constant_initialized_tensors = st->constant_initialized_tensors_;

              if (constant_initialized_tensors.count(ort_value_idx)) {
                bool is_packed = false;
                const Tensor& const_initialized_tensor = constant_initialized_tensors[ort_value_idx].Get<Tensor>();

                auto iter = initializers_to_share_map.find(input_name);
                bool is_shared_initializer = (iter != initializers_to_share_map.end());

                // Caching pre-packed weights is limited to shared initializers associated with the CPU EP for now
                if (is_shared_initializer && should_cache_prepacked_weights_for_shared_initializers &&
                    node.GetExecutionProviderType() == kCpuExecutionProvider) {  // caching of pre-packed weights' turned ON

                  AllocatorPtr allocator_for_caching = prepacked_weights_container_->GetOrCreateAllocator(CPU);
                  ORT_ENFORCE(allocator_for_caching.get() != nullptr);

                  PrePackedWeights weights_to_be_filled_in;
                  // The reason we invoke PrePack() before looking into the container for any pre-packed weight
                  // cached by another instance of the same op_type (for the same constant initializer) is because
                  // to truly know if we can use a cached pre-packed weight, we would have to compare the cached pre-packed
                  // weight with the pre-packed weight generated by this instance of the same op_type because other static
                  // properties of the node like node attributes could play a role in the pre-packed weights' contents.
                  ORT_RETURN_IF_ERROR(kernel->PrePack(const_initialized_tensor, input_idx, allocator_for_caching,
                                                      is_packed,
                                                      &weights_to_be_filled_in));

                  if (is_packed) {
                    // BUG CHECK: Ensure that the kernel has filled in the pre-packed weight to be cached if the weight was pre-packed
                    ORT_ENFORCE(weights_to_be_filled_in.buffers_.size() > 0, "The kernel corresponding to the node ", node.Name(),
                                " doesn't have an implementation that can cache computed pre-packed weights");

                    const auto& op_type = node.OpType();

                    // Sanity check
                    // TODO: Check if some version of the ONNX IR allows op_type to be empty
                    ORT_ENFORCE(!op_type.empty(), "The op type of a node cannot be empty");

                    // The key for the pre-packed weights container lookup is the op_type + hash of the prepacked-weight
                    // that we just got by invoking PrePack() on this kernel.

                    const std::string& prepacked_weights_container_key = GenerateKeyForPrepackedWeightsMap(op_type,
                                                                                                           weights_to_be_filled_in);

                    bool container_contains_packed_weight = prepacked_weights_container_->HasWeight(prepacked_weights_container_key);

                    if (container_contains_packed_weight) {
                      LOGS(logger_, INFO) << "Using cached version of pre-packed weight for constant initializer: " << input_name
                                          << " used in the node: " << node.Name() << " which is of op type: " << node.OpType();

                      ORT_RETURN_IF_ERROR(KernelUseSharedPrePackedBuffers(*kernel, input_idx,
                                                                          prepacked_weights_container_->GetWeight(prepacked_weights_container_key),
                                                                          node.Name()));

                      ++used_shared_pre_packed_weights_counter_;
                    } else {  // container doesn't contain the pre-packed weight - so write into it for sharing across kernel instances

                      if (!prepacked_weights_container_->WriteWeight(prepacked_weights_container_key, std::move(weights_to_be_filled_in))) {
                        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unable to write the provided PrePackedWeights instance into the container");
                      }

                      ORT_RETURN_IF_ERROR(KernelUseSharedPrePackedBuffers(*kernel, input_idx,
                                                                          prepacked_weights_container_->GetWeight(prepacked_weights_container_key),
                                                                          node.Name()));
                    }
                  }

                } else {  // caching of pre-packed weights' turned OFF
                  AllocatorPtr session_cpu_alloc = kernel->Info().GetAllocator(0, OrtMemType::OrtMemTypeDefault);
                  ORT_RETURN_IF_ERROR(kernel->PrePack(const_initialized_tensor, input_idx,
                                                      session_cpu_alloc,  // use allocator tied to this session
                                                      is_packed,
                                                      nullptr  // no caching required
                                                      ));
                }
                if (is_packed) {
                  ++number_of_prepacks_counter_;

                  if (constant_initializers_use_count.count(input_name) && --constant_initializers_use_count[input_name] == 0) {
                    // release the constant initialized tensor
                    st->initialized_tensors_.erase(ort_value_idx);
                    constant_initialized_tensors.erase(ort_value_idx);
                  }
                }
              }
              // stop searching in 2 cases:
              // 1. value is not from OuterScope
              // 2. value is from OuterScope and the current OuterScope has the value
              if (st != this || !st->graph_.IsOuterScopeValue(input_name)) {
                break;
              }
            }
            st = st->Parent();
          } while (st);
        }
        input_idx++;
      }
    }

    return Status::OK();
  };

  bool should_cache_prepacked_weights_for_shared_initializers = (prepacked_weights_container_ != nullptr);

  if (should_cache_prepacked_weights_for_shared_initializers) {
    // serialize calls to the method that looks up the container, calls UseCachedPrePackedWeight/PrePack
    // and writes pre-packed weights to the container
    std::lock_guard<onnxruntime::OrtMutex> l(prepacked_weights_container_->mutex_);
    return prepacked_constant_weights(true);
  } else {
    return prepacked_constant_weights(false);
  }
}

static int64_t CalculateMemoryPatternsKey(const gsl::span<const OrtValue>& tensor_inputs) {
  int64_t key = 0;
  for (const auto& input : tensor_inputs) {
    for (auto dim : input.Get<Tensor>().Shape().GetDims()) key ^= dim;
  }
  return key;
}

#ifdef ENABLE_TRAINING
namespace {
Status ResolveDimParams(const GraphViewer& graph,
                        const InlinedHashMap<std::string, TensorShape>& feeds,
                        InlinedHashMap<std::string, int64_t>& out) {
  out.reserve(graph.GetInputs().size());
  for (const auto* input : graph.GetInputs()) {
    auto* shape = input->Shape();
    auto it = feeds.find(input->Name());
    if (it == feeds.end()) {
      return Status(ONNXRUNTIME, FAIL,
                    "Graph input " + input->Name() +
                        " is not found in the feed list, unable to resolve the value for dynamic shape.");
    }
    if (it->second.NumDimensions() == 0 && !shape) {
      // This is a scalar, which has nothing to do with symbolic shapes.
      continue;
    }
    if (!shape || shape->dim_size() != static_cast<int>(it->second.NumDimensions())) {
      return Status(ONNXRUNTIME, FAIL, "Graph input " + input->Name() +
                                           "'s shape is not present or its shape doesn't match feed's shape."
                                           "Unable to resolve the value for dynamic shape");
    }
    for (int k = 0, end = shape->dim_size(); k < end; ++k) {
      if (shape->dim()[k].has_dim_param()) {
        out.insert({shape->dim()[k].dim_param(), it->second.GetDims()[k]});
      }
    }
  }
  return Status::OK();
}

Status TryResolveShape(
    const NodeArg* arg,
    const InlinedHashMap<std::string, int64_t>& symbolic_dimensions,
    size_t& is_resolved,  // indicate whether resolve successfully or not.
    TensorShapeVector& resolved_shape) {
  if (!arg->Shape()) {
    is_resolved = 0;
    return Status::OK();
  }

  TensorShapeVector shape;

  SafeInt<size_t> safe_size = 1;
  for (auto& dim : arg->Shape()->dim()) {
    if (dim.has_dim_param()) {
      auto it = symbolic_dimensions.find(dim.dim_param());
      if (it == symbolic_dimensions.end()) {
        return Status(ONNXRUNTIME, FAIL, "Unknown symbolic dimension, " + dim.dim_param() + ", found in memory pattern compute.");
      }
      safe_size *= it->second;
      shape.push_back(it->second);
    } else if (dim.has_dim_value() && dim.dim_value() > 0) {
      safe_size *= dim.dim_value();
      shape.push_back(dim.dim_value());
    } else {
      // tensor shape is unknown.
      safe_size = 0;
    }
  }

  is_resolved = safe_size;
  // Only assign shape if all symbolic dimensions are resolved.
  if (is_resolved != 0) {
    resolved_shape = std::move(shape);
  }

  return Status::OK();
}

void TryCalculateSizeFromResolvedShape(int ml_value_idx, const InlinedHashMap<int, TensorShape>& resolved_shapes, size_t& size) {
  size = 0;
  auto shape = resolved_shapes.find(ml_value_idx);
  if (shape != resolved_shapes.end()) {
    size = 1;
    for (auto dim : shape->second.GetDims())
      size *= dim;
  }
}

}  // namespace

// If this function fails NO memory planning will take place, hence lets ONLY FAIL and stop training where warranted, example SIZE overflow.
Status SessionState::GeneratePatternGroupCache(gsl::span<const OrtValue> tensor_inputs,
                                               gsl::span<const int> feed_mlvalue_idxs,
                                               MemoryPatternGroup& output,
                                               InlinedHashMap<int, TensorShape>& resolved_shapes) const {
  InlinedHashMap<std::string, TensorShape> feeds;
  feeds.reserve(feed_mlvalue_idxs.size());
  for (size_t i = 0, end = feed_mlvalue_idxs.size(); i < end; ++i) {
    std::string name;
    ORT_RETURN_IF_ERROR(this->ort_value_name_idx_map_.GetName(feed_mlvalue_idxs[i], name));
    feeds.emplace(std::move(name), tensor_inputs[i].Get<Tensor>().Shape());
  }
  InlinedHashMap<std::string, int64_t> map;
  ORT_RETURN_IF_ERROR(ResolveDimParams(*graph_viewer_, feeds, map));
  auto* exe_plan = GetExecutionPlan();
  ORT_ENFORCE(exe_plan);
  OrtValuePatternPlanner mem_planner(*exe_plan, /*using counters*/ true);

  // Try to resolve shapes for activations.
  auto& node_index_info = GetNodeIndexInfo();
  for (auto& node_plan : exe_plan->execution_plan) {
    int node_index = node_index_info.GetNodeOffset(node_plan.node_index);
    auto* node = graph_viewer_->GetNode(node_plan.node_index);
    int output_start = node_index + static_cast<int>(node->InputDefs().size()) +
                       static_cast<int>(node->ImplicitInputDefs().size());

    for (int i = 0, end = static_cast<int>(node->OutputDefs().size()); i < end; ++i) {
      const auto ml_value_idx = node_index_info.GetMLValueIndex(output_start + i);
      if (ml_value_idx == NodeIndexInfo::kInvalidEntry)
        continue;

      const auto* ml_type = exe_plan->allocation_plan[ml_value_idx].value_type;
      if (!ml_type->IsTensorType())
        continue;

      auto* arg = node->OutputDefs()[i];
      size_t is_resolved = 0;
      TensorShapeVector resolved_shape;

      // Tensors whose shape cannot be resolved statically will be allocated at runtime.
      if (TryResolveShape(arg, map, is_resolved, resolved_shape).IsOK()) {
        // Store all valid resolved shapes. They will be queried in, for example,
        // Recv operator to bypass the dependency of output shapes on inputs.
        if (is_resolved != 0) {
          resolved_shapes[ml_value_idx] = gsl::make_span(resolved_shape);
        }
      } else {
        LOGS(logger_, INFO) << "[Static memory planning] Could not resolve shape for tensor with ML index "
                            << ml_value_idx << ", will allocate dynamically.";
      }
    }
  }

  // Allocate activations that want to be laid out contiguously in memory.
  for (auto ml_value_idx : exe_plan->activation_allocation_order) {
    ORT_ENFORCE(ml_value_idx >= 0);

    const auto* ml_type = exe_plan->allocation_plan[ml_value_idx].value_type;
    if (!ml_type->IsTensorType())
      continue;
    const auto* ml_data_type = static_cast<const TensorTypeBase*>(ml_type)->GetElementType();
    if (exe_plan->allocation_plan[ml_value_idx].alloc_kind == AllocKind::kAllocate &&
        ml_data_type != DataTypeImpl::GetType<std::string>()) {
      size_t size = 0;
      TryCalculateSizeFromResolvedShape(ml_value_idx, resolved_shapes, size);
      if (size == 0) {
        std::string node_name;
        ORT_RETURN_IF_ERROR(this->ort_value_name_idx_map_.GetName(ml_value_idx, node_name));
        return Status(ONNXRUNTIME, FAIL, "Unknown shape found in memory pattern compute, node name is : " + node_name);
      }

      if (!IAllocator::CalcMemSizeForArrayWithAlignment<kAllocAlignment>(size, ml_data_type->Size(), &size)) {
        return Status(ONNXRUNTIME, FAIL, "Size overflow");
      }

      ORT_ENFORCE(exe_plan->allocation_plan[ml_value_idx].alloc_kind == AllocKind::kAllocate);

      const auto& counter = exe_plan->allocation_plan[ml_value_idx].program_counter;
      ORT_RETURN_IF_ERROR(mem_planner.TraceAllocation(ml_value_idx, counter, size));
    }
  }

  // Allocate all other activations.
  for (auto& node_plan : exe_plan->execution_plan) {
    int node_index = node_index_info.GetNodeOffset(node_plan.node_index);
    auto* node = graph_viewer_->GetNode(node_plan.node_index);
    int output_start = node_index + static_cast<int>(node->InputDefs().size()) +
                       static_cast<int>(node->ImplicitInputDefs().size());
    // allocate output
    for (int i = 0, end = static_cast<int>(node->OutputDefs().size()); i < end; ++i) {
      const auto ml_value_idx = node_index_info.GetMLValueIndex(output_start + i);
      if (ml_value_idx == NodeIndexInfo::kInvalidEntry ||
          (std::find(exe_plan->activation_allocation_order.begin(),
                     exe_plan->activation_allocation_order.end(), ml_value_idx) !=
           exe_plan->activation_allocation_order.end()))
        continue;
      const auto* ml_type = exe_plan->allocation_plan[ml_value_idx].value_type;
      if (!ml_type->IsTensorType())
        continue;
      const auto* ml_data_type = static_cast<const TensorTypeBase*>(ml_type)->GetElementType();
      size_t size = 0;
      TryCalculateSizeFromResolvedShape(ml_value_idx, resolved_shapes, size);

      // Plan memory if conditions are met.
      if (exe_plan->allocation_plan[ml_value_idx].alloc_kind == AllocKind::kAllocate &&
          ml_data_type != DataTypeImpl::GetType<std::string>() && size != 0) {
        size_t aligned_size = 0;
        if (!IAllocator::CalcMemSizeForArrayWithAlignment<kAllocAlignment>(size, ml_data_type->Size(), &aligned_size)) {
          return Status(ONNXRUNTIME, FAIL, "Size overflow");
        }

        ORT_ENFORCE(exe_plan->allocation_plan[ml_value_idx].alloc_kind == AllocKind::kAllocate);

        const auto& counter = exe_plan->allocation_plan[ml_value_idx].program_counter;
        ORT_RETURN_IF_ERROR(mem_planner.TraceAllocation(ml_value_idx, counter, aligned_size));
      }
    }

    // release nodes
    for (int index = node_plan.free_from_index; index <= node_plan.free_to_index; ++index) {
      auto ml_value_idx = exe_plan->to_be_freed[index];
      const auto* ml_type = exe_plan->allocation_plan[ml_value_idx].value_type;
      if (!ml_type->IsTensorType())
        continue;
      const auto* ml_data_type = static_cast<const TensorTypeBase*>(ml_type)->GetElementType();
      if (ml_data_type != DataTypeImpl::GetType<std::string>()) {
        ORT_RETURN_IF_ERROR(mem_planner.TraceFree(ml_value_idx));
      }
    }
  }

  if (!mem_planner.GeneratePatterns(output).IsOK()) {
    return Status(ONNXRUNTIME, FAIL, "Generate Memory Pattern failed");
  }
  return Status::OK();
}

#endif

// MemoryPatternGroup pointer is cached. It only inserted upon creation
// and is not updated if already present.
const MemoryPatternGroup* SessionState::GetMemoryPatternGroup(
    gsl::span<const OrtValue> tensor_inputs,
    gsl::span<const int> feed_mlvalue_idxs,
    const InlinedHashMap<int, TensorShape>*& out_inferred_shapes) const {
  out_inferred_shapes = nullptr;
  int64_t key = CalculateMemoryPatternsKey(tensor_inputs);
  std::lock_guard<OrtMutex> lock(mem_patterns_lock_);
  auto it = mem_patterns_.find(key);
  if (it == mem_patterns_.end()) {
#ifdef ENABLE_TRAINING
    MemoryPatternGroup mem_patterns;
    InlinedHashMap<int, TensorShape> inferred_shapes;
    if (GeneratePatternGroupCache(tensor_inputs, feed_mlvalue_idxs, mem_patterns, inferred_shapes).IsOK()) {
      auto patt_insert = mem_patterns_.insert_or_assign(key, std::move(mem_patterns));
      auto ptr = &patt_insert.first->second;
      auto shape_insert = shape_patterns_.insert_or_assign(key, std::move(inferred_shapes));
      out_inferred_shapes = &shape_insert.first->second;
      return ptr;
    }
#else
    ORT_UNUSED_PARAMETER(feed_mlvalue_idxs);
#endif
    return nullptr;
  }

  auto patt_hit = shape_patterns_.find(key);
  if (patt_hit != shape_patterns_.cend()) {
    out_inferred_shapes = &patt_hit->second;
  }
  return &it->second;
}

void SessionState::ResolveMemoryPatternFlag() {
  if (enable_mem_pattern_) {
    for (auto* input : graph_viewer_->GetInputs()) {
      if (!input->HasTensorOrScalarShape()) {
        enable_mem_pattern_ = false;
        break;
      }
    }

    // For subgraphs, the implicit inputs need to meet the same crieria
    // as the explicit inputs for memory pattern to be enabled
    if (graph_viewer_->IsSubgraph()) {
      const auto* parent_node = graph_viewer_->ParentNode();

      for (auto* implicit_input : parent_node->ImplicitInputDefs()) {
        if (!implicit_input->HasTensorOrScalarShape()) {
          enable_mem_pattern_ = false;
          break;
        }
      }
    }
  }
}

Status SessionState::UpdateMemoryPatternGroupCache(gsl::span<const OrtValue> tensor_inputs,
                                                   MemoryPatternGroup mem_patterns) const {
  int64_t key = CalculateMemoryPatternsKey(tensor_inputs);

  std::lock_guard<OrtMutex> lock(mem_patterns_lock_);
  // Do not update if present, as the pointer to the existing one is cached
  mem_patterns_.emplace(key, std::move(mem_patterns));
  return Status::OK();
}

bool SessionState::GetEnableMemoryPattern() const { return enable_mem_pattern_; }

bool SessionState::GetEnableMemoryReuse() const { return enable_mem_reuse_; }

common::Status SessionState::AddInputNameToNodeInfoMapping(const std::string& input_name, const NodeInfo& node_info) {
  // Graph partitioning should ensure an input is only consumed from one device. Copy nodes should have been inserted
  // to handle a scenario where an input is required on different devices by different nodes. Validate that.
  auto& entries = input_names_to_nodeinfo_mapping_[input_name];

  if (entries.empty()) {
    entries.push_back(node_info);
  } else {
    const auto& existing_entry = entries.front();

    // if index == max it's an entry for an implicit input to a subgraph or unused graph input.
    // we want to prefer the entry for explicit usage in this graph, as the implicit usage in a
    // subgraph will be handled by the subgraph's SessionState.
    if (node_info.index == std::numeric_limits<size_t>::max()) {
      // ignore and preserve existing value
    } else if (existing_entry.index == std::numeric_limits<size_t>::max()) {
      // replace existing entry that is for an implicit input with new entry for explicit usage in this graph
      entries[0] = node_info;
    } else {
      // if the devices match we can add the new entry for completeness (it will be ignored in
      // utils::CopyOneInputAcrossDevices though).
      // if they don't, we are broken.
      const auto& current_device = entries[0].device;
      const auto& new_device = node_info.device;

      if (current_device == new_device) {
        entries.push_back(node_info);
      } else {
        return ORT_MAKE_STATUS(
            ONNXRUNTIME, NOT_IMPLEMENTED,
            "Using an input in multiple nodes on different devices is not supported currently. Input:", input_name,
            " is used by node ", existing_entry.p_node->Name(), " (", current_device->ToString(), ") and node ",
            node_info.p_node->Name(), " (", new_device->ToString(), ").");
      }
    }
  }

  return Status::OK();
}

common::Status SessionState::GetInputNodeInfo(const std::string& input_name,
                                              InlinedVector<NodeInfo>& node_info_vec) const {
  auto entry = input_names_to_nodeinfo_mapping_.find(input_name);
  if (entry == input_names_to_nodeinfo_mapping_.cend()) {
    return Status(ONNXRUNTIME, FAIL, "Failed to find input name in the mapping: " + input_name);
  }

  node_info_vec = entry->second;
  return Status::OK();
}

const SessionState::NameNodeInfoMapType& SessionState::GetInputNodeInfoMap() const {
  return input_names_to_nodeinfo_mapping_;
}

void SessionState::AddOutputNameToNodeInfoMapping(const std::string& output_name, const NodeInfo& node_info) {
  auto& output_names_to_nodeinfo = output_names_to_nodeinfo_mapping_[output_name];
  ORT_ENFORCE(output_names_to_nodeinfo.empty(), "Only one node should produce an output. Existing entry for ",
              output_name);

  output_names_to_nodeinfo.push_back(node_info);
}

common::Status SessionState::GetOutputNodeInfo(const std::string& output_name,
                                               InlinedVector<NodeInfo>& node_info_vec) const {
  auto entry = output_names_to_nodeinfo_mapping_.find(output_name);
  if (entry == output_names_to_nodeinfo_mapping_.cend()) {
    return Status(ONNXRUNTIME, FAIL, "Failed to find output name in the mapping: " + output_name);
  }

  node_info_vec = entry->second;
  return Status::OK();
}

const SessionState::NameNodeInfoMapType& SessionState::GetOutputNodeInfoMap() const {
  return output_names_to_nodeinfo_mapping_;
}

void SessionState::AddSubgraphSessionState(onnxruntime::NodeIndex index, const std::string& attribute_name,
                                           std::unique_ptr<SessionState> session_state) {
  auto entry = subgraph_session_states_.find(index);

  // make sure this is new. internal logic error if it is not so using ORT_ENFORCE.
  if (entry != subgraph_session_states_.cend()) {
    const auto& existing_entries = entry->second;
    ORT_ENFORCE(existing_entries.find(attribute_name) == existing_entries.cend(), "Entry exists in node ", index,
                " for attribute ", attribute_name);
  }

  session_state->parent_ = this;

#ifdef ONNXRUNTIME_ENABLE_INSTRUMENT
  GenerateGraphId();
#endif

  subgraph_session_states_[index].insert(std::make_pair(attribute_name, std::move(session_state)));
}

SessionState* SessionState::GetMutableSubgraphSessionState(onnxruntime::NodeIndex index,
                                                           const std::string& attribute_name) {
  SessionState* session_state = nullptr;

  auto node_entry = subgraph_session_states_.find(index);
  if (node_entry != subgraph_session_states_.cend()) {
    const auto& attribute_state_map = node_entry->second;

    const auto& subgraph_entry = attribute_state_map.find(attribute_name);
    if (subgraph_entry != attribute_state_map.cend()) {
      session_state = subgraph_entry->second.get();
    }
  }

  return session_state;
}

const SessionState* SessionState::GetSubgraphSessionState(onnxruntime::NodeIndex index,
                                                          const std::string& attribute_name) const {
  return const_cast<SessionState*>(this)->GetMutableSubgraphSessionState(index, attribute_name);
}

const NodeIndexInfo& SessionState::GetNodeIndexInfo() const {
  ORT_ENFORCE(node_index_info_.has_value(), "SetGraphAndCreateKernels must be called prior to GetExecutionInfo.");
  return *node_index_info_;
}

#if !defined(ORT_MINIMAL_BUILD)
void SessionState::UpdateToBeExecutedNodes(gsl::span<int const> fetch_mlvalue_idxs) {
  InlinedVector<int> sorted_idxs;
  sorted_idxs.reserve(fetch_mlvalue_idxs.size());
  sorted_idxs.assign(fetch_mlvalue_idxs.begin(), fetch_mlvalue_idxs.end());
  std::sort(sorted_idxs.begin(), sorted_idxs.end());
  if (to_be_executed_nodes_.find(sorted_idxs) != to_be_executed_nodes_.end())
    return;

  // Get the nodes generating the fetches.
  InlinedVector<const Node*> nodes;
  nodes.reserve(fetch_mlvalue_idxs.size());
  InlinedHashSet<NodeIndex> reachable_nodes;
  reachable_nodes.reserve(graph_.NumberOfNodes());

  for (auto idx : fetch_mlvalue_idxs) {
    std::string node_arg_name;
    const auto status = this->GetOrtValueNameIdxMap().GetName(idx, node_arg_name);
    ORT_ENFORCE(status.IsOK(), status.ErrorMessage());
    auto ending_node = graph_.GetProducerNode(node_arg_name);
    nodes.push_back(ending_node);
  }

  // Reversely traverse to get reachable nodes.
  graph_.ReverseDFSFrom(
      nodes, {}, [&reachable_nodes](const Node* n) { reachable_nodes.insert(n->Index()); });
  to_be_executed_nodes_.emplace(std::move(sorted_idxs), std::move(reachable_nodes));
}

const InlinedHashSet<NodeIndex>* SessionState::GetToBeExecutedNodes(
    gsl::span<int const> fetch_mlvalue_idxs) const {
  InlinedVector<int> sorted_idxs;
  sorted_idxs.reserve(fetch_mlvalue_idxs.size());
  sorted_idxs.assign(fetch_mlvalue_idxs.begin(), fetch_mlvalue_idxs.end());
  std::sort(sorted_idxs.begin(), sorted_idxs.end());
  auto it = to_be_executed_nodes_.find(sorted_idxs);
  return (it != to_be_executed_nodes_.end()) ? &it->second : nullptr;
}

static Status GetSubGraphSessionStatesOrtFormat(
    flatbuffers::FlatBufferBuilder& builder,
    const SubgraphSessionStateMap& subgraph_session_states,
    std::vector<flatbuffers::Offset<fbs::SubGraphSessionState>>& fbs_subgraph_session_states) {
  size_t number_of_states = 0;
  for (const auto& pair : subgraph_session_states) {
    number_of_states += pair.second.size();
  }
  fbs_subgraph_session_states.clear();
  fbs_subgraph_session_states.reserve(number_of_states);
  for (const auto& [node_idx, session_states] : subgraph_session_states) {
    for (const auto& name_to_subgraph_session_state : session_states) {
      const std::string& attr_name = name_to_subgraph_session_state.first;
      SessionState& subgraph_session_state = *name_to_subgraph_session_state.second;
      auto graph_id = builder.CreateString(fbs::utils::GetSubgraphId(node_idx, attr_name));
      flatbuffers::Offset<fbs::SessionState> session_state;
      ORT_RETURN_IF_ERROR(
          subgraph_session_state.SaveToOrtFormat(builder, session_state));

      fbs_subgraph_session_states.push_back(
          fbs::CreateSubGraphSessionState(builder, graph_id, session_state));
    }
  }
  return Status::OK();
}

Status SessionState::SaveToOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                                     flatbuffers::Offset<fbs::SessionState>& fbs_session_state) const {
  size_t size = kernel_create_info_map_.size();
  std::vector<uint32_t> node_indices;
  std::vector<uint64_t> kernel_def_hashes;
  node_indices.reserve(size);
  kernel_def_hashes.reserve(size);
  for (const auto& kvp : kernel_create_info_map_) {
    node_indices.push_back(gsl::narrow<uint32_t>(kvp.first));
    kernel_def_hashes.push_back(kvp.second->kernel_def->GetHash());
  }

  auto kernels = fbs::CreateKernelCreateInfosDirect(builder, &node_indices, &kernel_def_hashes);

  // Subgraph session states
  std::vector<flatbuffers::Offset<fbs::SubGraphSessionState>> sub_graph_session_states;
  ORT_RETURN_IF_ERROR(
      GetSubGraphSessionStatesOrtFormat(builder, subgraph_session_states_, sub_graph_session_states));

  fbs_session_state = fbs::CreateSessionStateDirect(builder, kernels, &sub_graph_session_states);
  return Status::OK();
}

#endif  // !defined(ORT_MINIMAL_BUILD)

Status SessionState::CreateSubgraphSessionState() {
  for (auto& node : graph_.Nodes()) {
    for (auto& entry : node.GetAttributeNameToMutableSubgraphMap()) {
      const auto& ep = node.GetExecutionProviderType();
      if (!ep.empty() && ep != kCpuExecutionProvider && ep != kCudaExecutionProvider) {
        // SessionState is only used when ORT is executing the subgraph. If a non-ORT EP has taken the control flow
        // node containing the subgraph it will create whatever state it needs internally.
        continue;
      }

      auto& attr_name = entry.first;
      Graph* subgraph = entry.second;
      ORT_ENFORCE(subgraph, "Main Graph instance should have populated all subgraphs when being resolved.");

      auto subgraph_session_state =
          std::make_unique<SessionState>(*subgraph, execution_providers_, enable_mem_pattern_,
                                         thread_pool_, inter_op_thread_pool_, data_transfer_mgr_,
                                         logger_, profiler_);

      // Pass fused function manager to subgraph
      subgraph_session_state->fused_funcs_mgr_.SetFusedFuncs(fused_funcs_mgr_);

      // recurse
      ORT_RETURN_IF_ERROR(subgraph_session_state->CreateSubgraphSessionState());

      // add the subgraph SessionState instance to the parent graph SessionState so it can be retrieved
      // by Compute() via OpKernelContextInternal.
      AddSubgraphSessionState(node.Index(), attr_name, std::move(subgraph_session_state));
    }
  }

  return Status::OK();
}

Status SessionState::LoadFromOrtFormat(const fbs::SessionState& fbs_session_state,
                                       const KernelRegistryManager& kernel_registry_manager) {
  using fbs::utils::FbsSessionStateViewer;
  const FbsSessionStateViewer fbs_session_state_viewer{fbs_session_state};
  ORT_RETURN_IF_ERROR(fbs_session_state_viewer.Validate());

  // look up KernelCreateInfo with hash and
  // - add KernelCreateInfo for node
  // - set node's EP from KernelCreateInfo if unset
  auto add_kernel_and_set_node_ep_by_hash =
      [&kernel_registry_manager, this](Node& node, HashValue hash) {
        const KernelCreateInfo* kci = nullptr;
        utils::UpdateHashForBackwardsCompatibility(hash);

        ORT_RETURN_IF_NOT(kernel_registry_manager.SearchKernelRegistriesByHash(hash, &kci),
                          "Failed to find kernel def hash (", hash, ") in kernel registries for ",
                          node.OpType(), "(", node.SinceVersion(), ") node with name '", node.Name(), "'.");

        {
          const auto [it, inserted] = kernel_create_info_map_.emplace(node.Index(),
                                                                      gsl::not_null<const KernelCreateInfo*>(kci));
          ORT_RETURN_IF_NOT(inserted,
                            "Cannot overwrite existing kernel for ",
                            node.OpType(), "(", node.SinceVersion(), ") node with name '", node.Name(),
                            "'. Existing kernel def hash: ", it->second->kernel_def->GetHash(),
                            ", new kernel def hash: ", hash, ".");
        }

        if (node.GetExecutionProviderType().empty()) {
          node.SetExecutionProviderType(kci->kernel_def->Provider());
        } else {
          ORT_RETURN_IF_NOT(node.GetExecutionProviderType() == kci->kernel_def->Provider(),
                            "Node execution provider type mismatch. Existing: ", node.GetExecutionProviderType(),
                            ", from KernelCreateInfo (via hash lookup): ", kci->kernel_def->Provider());
        }

        return Status::OK();
      };

  // kernel hashes for model are in top level SessionState
  const auto& compiled_kernel_hashes = GetCompiledKernelHashes();

  // process the nodes that existed when the model was created
  for (FbsSessionStateViewer::Index i = 0, end = fbs_session_state_viewer.GetNumNodeKernelInfos(); i < end; ++i) {
    const auto node_kernel_info = fbs_session_state_viewer.GetNodeKernelInfo(i);

    Node* const node = graph_.GetNode(node_kernel_info.node_index);
    if (node == nullptr) {
#if defined(ORT_MINIMAL_BUILD) && !defined(ORT_EXTENDED_MINIMAL_BUILD)
      // this is OK if we have compiled kernels and the original node was replaced.
      // if not the model is invalid.
      ORT_RETURN_IF(compiled_kernel_hashes.empty(),
                    "Can't find node with index ", node_kernel_info.node_index, ". Invalid ORT format model.");
#endif  // defined(ORT_MINIMAL_BUILD) && !defined(ORT_EXTENDED_MINIMAL_BUILD)
      continue;
    }

    ORT_RETURN_IF_ERROR(add_kernel_and_set_node_ep_by_hash(*node, node_kernel_info.kernel_def_hash));
  }

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  // process the nodes that were added by replaying any loaded runtime optimizations
  for (const auto& [node_index, kernel_def_hash] :
       graph_.RuntimeOptimizationReplayCtx().produced_node_index_to_kernel_def_hash) {
    auto* node = graph_.GetNode(node_index);

    // NHWC optimizer may replace a node, so a missing node isn't necessarily an error
    // ORT_RETURN_IF(node == nullptr, "Can't find runtime optimization produced node with index ", node_index);

    if (node != nullptr) {
      ORT_RETURN_IF_ERROR(add_kernel_and_set_node_ep_by_hash(*node, kernel_def_hash));
    }
  }

  // Look up the hashes for any nodes we compiled or added during graph partitioning or other runtime optimizations.
  // These nodes are not in the original model as they were created at runtime.
  for (auto& node : graph_.Nodes()) {
    if (kernel_create_info_map_.find(node.Index()) != kernel_create_info_map_.end()) {
      continue;
    }

    if (node.Domain() == kOnnxDomain || node.Domain() == kMSDomain) {
      // two possible places to get hash from
      auto kernel_hash = utils::GetHashValueFromStaticKernelHashMap(node.OpType(), node.SinceVersion());
      if (!kernel_hash.has_value()) {
        kernel_hash = utils::GetInternalNhwcOpHash(node);
      }
      ORT_RETURN_IF_NOT(kernel_hash.has_value(),
                        "Unable to find kernel hash for node: '", node.Name(), "' optype: ", node.OpType());

      ORT_RETURN_IF_ERROR(add_kernel_and_set_node_ep_by_hash(node, *kernel_hash));
    } else {
      const auto hash_info = compiled_kernel_hashes.find(node.OpType());
      ORT_RETURN_IF(hash_info == compiled_kernel_hashes.cend(),
                    "Unable to find compiled kernel hash for node '", node.Name(), "'.");

      ORT_RETURN_IF_ERROR(add_kernel_and_set_node_ep_by_hash(node, hash_info->second));
    }
  }
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

  if (!subgraph_session_states_.empty()) {
    for (const auto& [node_idx, session_states] : subgraph_session_states_) {
      for (const auto& [attr_name, subgraph_session_state] : session_states) {
        const fbs::SessionState* fbs_subgraph_session_state;
        ORT_RETURN_IF_ERROR(fbs_session_state_viewer.GetSubgraphSessionState(node_idx, attr_name, fbs_subgraph_session_state));

        ORT_RETURN_IF_ERROR(subgraph_session_state->LoadFromOrtFormat(*fbs_subgraph_session_state, kernel_registry_manager));
      }
    }
  }

  return Status::OK();
}

// Calculate the use count of a constant initialized tensor, including the use in subgraph.
// Note: This function doesn't handle the case below:
// The main graph has a constant initializer called X, and the subgraph also has a constant initializer called X, which overrides the X from main graph.
// For case like this, the current implementation will calculate the use count as 2, but they could contain completely different values so each should have a use count of 1.
// This is a very rare case. If it happens and X is prepacked, the consequence is that X won't be released and memory usage of X won't be saved. This will be fine.
static void ComputeConstantInitializerUseCount(const Graph& graph, InlinedHashMap<std::string, size_t>& constant_initializers_use_count) {
  for (const auto& node : graph.Nodes()) {
    for (const auto* arg : node.InputDefs()) {
      if (arg->Exists() && graph.GetConstantInitializer(arg->Name(), true /*check_outer_scope*/)) {
        constant_initializers_use_count[arg->Name()]++;
      }
    }

    if (node.ContainsSubgraph()) {
      for (const gsl::not_null<const Graph*>& subgraph : node.GetSubgraphs()) {
        ComputeConstantInitializerUseCount(*subgraph, constant_initializers_use_count);
      }
    }
  }
  // Initializers can be used as graph outputs
  for (const auto* arg : graph.GetOutputs()) {
    if (arg->Exists() && graph.GetConstantInitializer(arg->Name(), true /*check_outer_scope*/)) {
      constant_initializers_use_count[arg->Name()]++;
    }
  }
}

using NodePlacementMap = std::unordered_map<std::string, std::vector<std::string>>;

static Status VerifyEachNodeIsAssignedToAnEpImpl(const Graph& graph, bool is_verbose,
                                                 NodePlacementMap& node_placements) {
  for (const auto& node : graph.Nodes()) {
    const auto& node_provider = node.GetExecutionProviderType();
    if (node_provider.empty()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                             "Could not find an implementation for ",
                             node.OpType(), "(", node.SinceVersion(), ") node with name '", node.Name(), "'");
    }

#if !defined(ORT_MINIMAL_BUILD)
    if (is_verbose) {  // TODO: should we disable this if the number of nodes is above a certain threshold?
      const std::string node_str = node.OpType() + " (" + node.Name() + ")";
      node_placements[node_provider].push_back(node_str);
    }
#endif  // !defined(ORT_MINIMAL_BUILD)

    // recurse into subgraphs
    if (node.ContainsSubgraph()) {
      const auto subgraphs = node.GetSubgraphs();
      for (const auto& subgraph : subgraphs) {
        ORT_RETURN_IF_ERROR(VerifyEachNodeIsAssignedToAnEpImpl(*subgraph, is_verbose, node_placements));
      }
    }
  }

  return Status::OK();
}

static Status VerifyEachNodeIsAssignedToAnEp(const Graph& graph, const logging::Logger& logger) {
  NodePlacementMap node_placements{};
#if !defined(ORT_MINIMAL_BUILD)
  const bool is_verbose_mode = logger.GetSeverity() == logging::Severity::kVERBOSE;
#else
  ORT_UNUSED_PARAMETER(logger);
  const bool is_verbose_mode = false;
#endif  // !defined(ORT_MINIMAL_BUILD)

  ORT_RETURN_IF_ERROR(VerifyEachNodeIsAssignedToAnEpImpl(graph, is_verbose_mode, node_placements));

#if !defined(ORT_MINIMAL_BUILD)
  // print placement info
  if (is_verbose_mode) {
    LOGS(logger, VERBOSE) << "Node placements";
    if (node_placements.size() == 1) {
      const auto& [provider, node_strs] = *node_placements.begin();
      LOGS(logger, VERBOSE) << " All nodes placed on [" << provider << "]. Number of nodes: " << node_strs.size();
    } else {
      for (const auto& [provider, node_strs] : node_placements) {
        LOGS(logger, VERBOSE) << " Node(s) placed on [" << provider << "]. Number of nodes: " << node_strs.size();
        for (const auto& node_str : node_strs) {
          LOGS(logger, VERBOSE) << "  " << node_str;
        }
      }
    }
  }
#endif  // !defined(ORT_MINIMAL_BUILD)

  return Status::OK();
}

Status SessionState::FinalizeSessionState(const std::basic_string<PATH_CHAR_TYPE>& graph_location,
                                          const KernelRegistryManager& kernel_registry_manager,
                                          const SessionOptions& session_options,
                                          const onnxruntime::fbs::SessionState* serialized_session_state,
                                          bool remove_initializers,
                                          bool saving_ort_format) {
  // recursively create the subgraph session state instances and populate the kernel create info in them.
  // it's simpler to handle the kernel create info recursively when deserializing,
  // so also do it recursively when calling PopulateKernelCreateInfo for consistency.
  ORT_RETURN_IF_ERROR(CreateSubgraphSessionState());

  if (serialized_session_state) {
    ORT_RETURN_IF_ERROR(LoadFromOrtFormat(*serialized_session_state, kernel_registry_manager));
    // LoadFromOrtFormat() may assign node EPs so check afterwards
    ORT_RETURN_IF_ERROR(VerifyEachNodeIsAssignedToAnEp(graph_, logger_));
  } else {
#if !defined(ORT_MINIMAL_BUILD)
    ORT_RETURN_IF_ERROR(VerifyEachNodeIsAssignedToAnEp(graph_, logger_));
    ORT_RETURN_IF_ERROR(PopulateKernelCreateInfo(kernel_registry_manager, saving_ort_format));
#else
    ORT_UNUSED_PARAMETER(graph_location);
    ORT_UNUSED_PARAMETER(kernel_registry_manager);
    ORT_UNUSED_PARAMETER(session_options);
    ORT_UNUSED_PARAMETER(remove_initializers);
    ORT_UNUSED_PARAMETER(saving_ort_format);
    return Status(ONNXRUNTIME, INVALID_ARGUMENT,
                  "Serialized session state must be provided from an ORT format model in this build.");
#endif
  }

  InlinedHashMap<std::string, size_t> constant_initializers_use_count;
  ComputeConstantInitializerUseCount(graph_, constant_initializers_use_count);
  return FinalizeSessionStateImpl(graph_location, kernel_registry_manager, nullptr, session_options,
                                  remove_initializers, constant_initializers_use_count);
}

static Status Index(const OrtValueNameIdxMap& ort_value_name_idx_map,
                    const OrtValueName& name,
                    /*out*/ OrtValueIndex& value) {
  return ort_value_name_idx_map.GetIdx(name, value);
}

static bool IsNodeWhereNodeInputsAreSameAsExplicitSubgraphInputs(const Node& node) {
  const auto& op_type = node.OpType();
  int since_version = node.SinceVersion();

  // TODO: Re-visit this method if more subgraph ops get accepted into ONNX

  // At the time of writing, there are only 3 ops in ONNX that have subgraphs
  // 1) If
  // 2) Loop
  // 3) Scan

  // `If` - The op doesn't have explicit subgraph inputs (return false)
  // `Loop`- In all opset versions of Loop (at the time of writing) the node inputs
  // have a one-to-one mapping between them and the explicit subgraph inputs
  // of the subgraph held in the Loop (return true)
  // `Scan` - Except opset 8 version of Scan (at the time of writing), all other
  // versions have the same one-to-one mapping as Loop (return true for opset > 8)

  return (op_type == "Loop" || (op_type == "Scan" && since_version >= 9));
}

// The following method accumulates the locations of all inputs (implicit and explicit)
// to a control flow node at the current graph level. This information will be used in
// the allocation planner while determining the location of such inputs in the subgraph.
// This method will not be called for the main graph (there is no concept of "outer scope" for the main graph).
static Status OuterScopeNodeArgLocationAccumulator(const SequentialExecutionPlan& plan,
                                                   const OrtValueNameIdxMap& ort_value_name_to_idx_map,
                                                   const Node& parent_node,
                                                   const GraphViewer& subgraph,
                                                   /*out*/ InlinedHashMap<OrtValueName, OrtMemoryInfo>& outer_scope_arg_to_location_map) {
  // Process implicit inputs to the node
  outer_scope_arg_to_location_map.reserve(parent_node.ImplicitInputDefs().size() + parent_node.InputDefs().size());
  auto process_implicit_input = [&plan, &ort_value_name_to_idx_map,
                                 &outer_scope_arg_to_location_map](const NodeArg& input, size_t /*arg_idx*/) {
    const auto& name = input.Name();
    OrtValueIndex index = -1;
    ORT_RETURN_IF_ERROR(Index(ort_value_name_to_idx_map, name, index));
    outer_scope_arg_to_location_map.insert({name, plan.GetLocation(index)});
    return Status::OK();
  };

  ORT_RETURN_IF_ERROR(Node::ForEachWithIndex(parent_node.ImplicitInputDefs(), process_implicit_input));

  // Process explicit inputs to the node
  // (they are passed through as explicit subgraph inputs and hence requires a re-mapping of names
  // to their corresponding names in the inner nested subgraph(s) held by the node)
  const auto& subgraph_inputs = subgraph.GetInputs();

  auto process_input = [&plan, &ort_value_name_to_idx_map, &outer_scope_arg_to_location_map,
                        &subgraph_inputs](const NodeArg& input, size_t arg_idx) {
    const auto& name = input.Name();
    OrtValueIndex index = -1;
    ORT_RETURN_IF_ERROR(Index(ort_value_name_to_idx_map, name, index));

    // Store the location of the outer scope value in the map using the subgraph input as the key
    // as that will be the referenced name in the subgraph (i.e.) re-mapping of names is required
    outer_scope_arg_to_location_map.insert({subgraph_inputs[arg_idx]->Name(), plan.GetLocation(index)});

    return Status::OK();
  };

  if (IsNodeWhereNodeInputsAreSameAsExplicitSubgraphInputs(parent_node)) {
    return Node::ForEachWithIndex(parent_node.InputDefs(), process_input);
  }

  return Status::OK();
}

// We accumulate all nested subgraph(s) kernel create info maps relative to the current depth
// (i.e.) if we were on the first nested subgraph, we accumulate information from ALL the
// nested subgraphs within it.
// This information is necessary to plan the right location for initializers
// in a given level because they could be used in one of the nested subgraphs relative to the
// current level (not just within the same level or even one level deep).
// Since we need to package up information from multiple levels of nested subgraphs, the key we use
// is "{key_for_node_containing_subgraph} + current_depth + node_index_containing_the_subgraph + attribute_name".
// {key_for_node_containing_subgraph} is empty for the main graph.

// For example, if we want to store information corresponding to a nested subgraph wrt to the main graph and
// the node index  of the node in the main graph was 2 and the attribute containing the specific
// subgraph was "then_branch", the key would be depth + node_index + attribute = 0 + 2 + then_branch
// = "02then_branch".

// If that subgraph contained another subgraph at node index 1, then the key would be,
// {02then_branch} + 1 + 1 + "then_branch" = "02then_branch11then_branch".

static void AccumulateAllNestedSubgraphsInfo(
    const SessionState& session_state,
    const std::string& subgraph_kernel_create_info_map_key_base,
    size_t graph_depth,
    /*out*/ SubgraphsKernelCreateInfoMaps& subgraphs_kernel_create_info_maps) {
  for (const auto& entry : session_state.GetSubgraphSessionStateMap()) {
    auto node_index = entry.first;

    for (const auto& name_to_subgraph_session_state : entry.second) {
      const auto& subgraph_attr_name = name_to_subgraph_session_state.first;

      SessionState& subgraph_session_state = *name_to_subgraph_session_state.second;

      const auto& local_subgraph_kernel_create_info_map_key =
          NestedSubgraphInfoDetails::ComposeNestedSubgraphInfoKeyHelper(subgraph_kernel_create_info_map_key_base,
                                                                        graph_depth, node_index, subgraph_attr_name);

      // The end user is never likely to see an error with the following line.
      // Points to an internal processing error if we hit this.
      ORT_ENFORCE(subgraphs_kernel_create_info_maps.find(local_subgraph_kernel_create_info_map_key) ==
                  subgraphs_kernel_create_info_maps.end());

      subgraphs_kernel_create_info_maps.insert({local_subgraph_kernel_create_info_map_key,
                                                subgraph_session_state.GetKernelCreateInfoMap()});

      // Recurse into the subgraph session state
      AccumulateAllNestedSubgraphsInfo(subgraph_session_state,
                                       local_subgraph_kernel_create_info_map_key,
                                       graph_depth + 1, subgraphs_kernel_create_info_maps);
    }
  }
}

Status SessionState::FinalizeSessionStateImpl(const std::basic_string<PATH_CHAR_TYPE>& graph_location,
                                              const KernelRegistryManager& kernel_registry_manager,
                                              _In_opt_ const Node* parent_node,
                                              const SessionOptions& session_options,
                                              bool remove_initializers,
                                              InlinedHashMap<std::string, size_t>& constant_initializers_use_count,
                                              const InlinedHashMap<OrtValueName, OrtMemoryInfo>& outer_scope_node_arg_to_location_map,
                                              bool graph_info_already_created) {
  if (!graph_info_already_created) {
    CreateGraphInfo();
  }

#if defined(ORT_EXTENDED_MINIMAL_BUILD)
  // Remove any unused initializers.
  // Not needed in a full build because unused initializers should have been removed earlier by Graph::Resolve().
  // Not needed in a basic minimal build because only runtime optimizations are expected to possibly result in unused
  //   initializers and they are only enabled in an extended minimal build.
  {
    InlinedVector<std::reference_wrapper<const std::string>> unused_initializer_names;
    unused_initializer_names.reserve(graph_.GetAllInitializedTensors().size());
    for (const auto& [name, tensor_proto] : graph_.GetAllInitializedTensors()) {
      ORT_UNUSED_PARAMETER(tensor_proto);
      int idx;
      if (!ort_value_name_idx_map_.GetIdx(name, idx).IsOK()) {
        unused_initializer_names.push_back(name);
      }
    }

    for (const auto& name : unused_initializer_names) {
      graph_.RemoveInitializedTensor(name);
    }
  }
#endif  // defined(ORT_EXTENDED_MINIMAL_BUILD)

  // ignore any outer scope args we don't know about. this can happen if a node contains multiple subgraphs.
  InlinedVector<const NodeArg*> valid_outer_scope_node_args;
  if (parent_node) {
    auto outer_scope_node_args = parent_node->ImplicitInputDefs();
    valid_outer_scope_node_args.reserve(outer_scope_node_args.size());

    std::for_each(outer_scope_node_args.cbegin(), outer_scope_node_args.cend(),
                  [this, &valid_outer_scope_node_args](const NodeArg* node_arg) {
                    int idx;
                    if (ort_value_name_idx_map_.GetIdx(node_arg->Name(), idx).IsOK()) {
                      valid_outer_scope_node_args.push_back(node_arg);
                    };
                  });
  }

  SubgraphsKernelCreateInfoMaps subgraphs_kernel_create_info_maps;
  AccumulateAllNestedSubgraphsInfo(*this, "", 0, subgraphs_kernel_create_info_maps);

  SequentialPlannerContext context(session_options.execution_mode, session_options.execution_order, session_options.enable_mem_reuse);
  ORT_RETURN_IF_ERROR(SequentialPlanner::CreatePlan(parent_node, *graph_viewer_, valid_outer_scope_node_args,
                                                    execution_providers_, kernel_create_info_map_,
                                                    subgraphs_kernel_create_info_maps,
                                                    outer_scope_node_arg_to_location_map,
                                                    ort_value_name_idx_map_, context, p_seq_exec_plan_));
// Record the allocation plan

// Uncomment the below to dump the allocation plan to std::cout
// LOGS(logger_, VERBOSE) << std::make_pair(p_seq_exec_plan_.get(), this);
#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
  GetMemoryProfiler()->Init(GetExecutionPlan(), GetOrtValueNameIdxMap());
#endif

  // Memory pattern tracer allocates all initializers on a single contiguous
  // buffer. This has the effect of reducing memory fragmentation.
  // Further more, NCCL kernels require initializers to be allocated
  // contiguously.
  //
  // In inferencing scenarios, however, we often want to pre-process and then
  // release some initializers. See OpKernel::PrePack(). Letting all initializers
  // sharing a single buffer makes it hard to release individual ones, leading
  // to memory waste.
  //
  // TODO!! disabling memory pattern tracer increases fragementation, leading to
  //  out of memory error in some training tests. Need to create kernel first,
  //  and let the kernel tells us whether the initalizer needs to be traced.
  //
#if defined(ENABLE_TRAINING)
  std::unique_ptr<ITensorAllocator> tensor_allocator(
      ITensorAllocator::Create(enable_mem_pattern_, *p_seq_exec_plan_, *this, weights_buffers_));
#else
  std::unique_ptr<ITensorAllocator> tensor_allocator(
      ITensorAllocator::Create(false, *p_seq_exec_plan_, *this, weights_buffers_));
#endif

  const auto& initializer_allocation_order = p_seq_exec_plan_->initializer_allocation_order;

  // move initializers from TensorProto instances in Graph to OrtValue instances in SessionState
  session_state_utils::MemoryProfileFunction memory_profile_func = nullptr;
#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
  memory_profile_func = [this](ITensorAllocator& planner) {
    GetMemoryProfiler()->GetMemoryInfo().RecordPatternInfo(
        planner.GetMemPatterns(), MemoryInfo::MapType::Initializer);
    GetMemoryProfiler()->CreateEvents(
        "initializer_" + std::to_string(GetMemoryProfiler()->GetMemoryInfo().GetIteration()),
        GetMemoryProfiler()->GetAndIncreasePid(), MemoryInfo::MapType::Initializer, "", 0);
  };

#endif

  ORT_RETURN_IF_ERROR(
      session_state_utils::SaveInitializedTensors(
          Env::Default(), graph_location, *graph_viewer_,
          execution_providers_.GetDefaultCpuAllocator(),
          ort_value_name_idx_map_, initializer_allocation_order, *tensor_allocator,
          [this, remove_initializers](const std::string& name, int idx, const OrtValue& value, const OrtCallback& d,
                                      bool constant, bool sparse) -> Status {
            ORT_RETURN_IF_ERROR(AddInitializedTensor(idx, value, &d, constant, sparse));
            if (remove_initializers) {
              graph_.RemoveInitializedTensor(name);
            }
            return Status::OK();
          },
          logger_, data_transfer_mgr_, *p_seq_exec_plan_, session_options, memory_profile_func));

#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
  // Record Weight allocation info on device
  GetMemoryProfiler()->GetMemoryInfo().RecordInitializerAllocInfo(GetInitializedTensors());
#endif

  // remove weights from the graph now to save memory but in many cases it won't save memory, if the tensor was
  // preallocated with the some other tensors in a single 'allocate' call, which is very common.
  // TODO: make it better
  if (remove_initializers) {
    CleanInitializedTensorsFromGraph();
  }

  ORT_RETURN_IF_ERROR(CreateKernels(kernel_registry_manager));

#ifndef ENABLE_TRAINING
  const auto disable_prepacking =
      session_options.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigDisablePrepacking, "0");

  if (disable_prepacking != "1") {
    ORT_RETURN_IF_ERROR(PrepackConstantInitializedTensors(constant_initializers_use_count,
                                                          session_options.initializers_to_share_map));
  }
#endif

  ORT_RETURN_IF_ERROR(
      session_state_utils::SaveInputOutputNamesToNodeMapping(*graph_viewer_, *this, valid_outer_scope_node_args));

  // Need to recurse into subgraph session state instances to finalize them and add the execution info

  // Currently all subgraphs need to be executed using the sequential EP due to potential deadlock with the current
  // parallel executor implementation
  SessionOptions subgraph_session_options(session_options);
  subgraph_session_options.execution_mode = ExecutionMode::ORT_SEQUENTIAL;

  for (const auto& node_to_subgraph_ss : subgraph_session_states_) {
    Node& node = *graph_.GetNode(node_to_subgraph_ss.first);

    for (const auto& attr_subgraph_pair : node.GetAttributeNameToMutableSubgraphMap()) {
      auto& attr_name = attr_subgraph_pair.first;
      auto entry = node_to_subgraph_ss.second.find(attr_name);
      // CreateSubgraphSessionState should ensure all these entries are created
      ORT_ENFORCE(entry != node_to_subgraph_ss.second.cend(),
                  "Missing session state for subgraph. Node:'", node.Name(),
                  "' OpType:", node.OpType(), " Index:", node.Index(), " Attribute:", attr_name);

      SessionState& subgraph_session_state = *entry->second;

      // recurse

      // We need to create graph info for the subgraphs because information accumulated there
      // is used in OuterScopeNodeArgLocationAccumulator()
      subgraph_session_state.CreateGraphInfo();

      InlinedHashMap<OrtValueName, OrtMemoryInfo> subgraph_outer_scope_node_arg_to_location_map;
      ORT_RETURN_IF_ERROR(OuterScopeNodeArgLocationAccumulator(*p_seq_exec_plan_, GetOrtValueNameIdxMap(),
                                                               node,
                                                               subgraph_session_state.GetGraphViewer(),
                                                               subgraph_outer_scope_node_arg_to_location_map));
      ORT_RETURN_IF_ERROR(subgraph_session_state.FinalizeSessionStateImpl(
          graph_location, kernel_registry_manager, &node, subgraph_session_options, remove_initializers,
          constant_initializers_use_count, subgraph_outer_scope_node_arg_to_location_map, true));

      // setup all the info for handling the feeds and fetches used in subgraph execution
      auto* p_op_kernel = GetMutableKernel(node.Index());
      ORT_ENFORCE(p_op_kernel);

      // Downcast is safe, since only control flow nodes have subgraphs
      // (node.GetAttributeNameToMutableSubgraphMap() is non-empty)
      auto& control_flow_kernel = static_cast<controlflow::IControlFlowKernel&>(*p_op_kernel);
      ORT_RETURN_IF_ERROR(control_flow_kernel.SetupSubgraphExecutionInfo(*this, attr_name, subgraph_session_state));
    }

    // TODO: Once the subgraph session states have been finalized, can we go back and plan the location of implicit
    // inputs that are fed through as graph inputs in the graph level holding the subgraphs ? Ideally the planned
    // locations for these would be the locations they are explicitly consumed on in nested subgraphs.
  }

  return Status::OK();
}

}  // namespace onnxruntime
