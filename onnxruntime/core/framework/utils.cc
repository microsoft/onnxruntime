// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/graph/onnx_protobuf.h"
#include "core/framework/utils.h"

#include <iomanip>

#include "core/graph/graph_viewer.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/execution_frame.h"
#include "core/framework/execution_providers.h"
#include "core/framework/feeds_fetches_manager.h"
#include "core/framework/kernel_def_builder.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/parallel_executor.h"
#include "core/framework/session_state.h"
#include "core/framework/sequential_executor.h"
#include "core/framework/tensorprotoutils.h"
#include "core/mlas/inc/mlas.h"
#include "core/framework/TensorSeq.h"
#ifdef ENABLE_TRAINING
#include "core/framework/orttraining_partial_executor.h"
#endif

#ifdef ENABLE_ATEN
#include "contrib_ops/cpu/aten_ops/aten_op_executor.h"
#endif

namespace onnxruntime {
namespace utils {
void* DefaultAlloc(size_t size) {
  return onnxruntime::AllocatorDefaultAlloc(size);
}

void DefaultFree(void* p) {
  onnxruntime::AllocatorDefaultFree(p);
}

void ConstructStrings(void* p_data, int64_t elements) {
  auto* ptr = static_cast<std::string*>(p_data);
  for (int64_t i = 0; i < elements; ++i) {
    new (ptr + i) std::string();
  }
}

void DestroyStrings(void* p_data, int64_t elements) {
  using string = std::string;
  auto* ptr = static_cast<std::string*>(p_data);
  for (int64_t i = 0; i < elements; i++)
    ptr[i].~string();
}

bool ProviderIsCpuBased(const std::string& provider_type) {
  return provider_type == onnxruntime::kCpuExecutionProvider ||
         provider_type == onnxruntime::kDnnlExecutionProvider ||
         provider_type == onnxruntime::kTvmExecutionProvider ||
         provider_type == onnxruntime::kVitisAIExecutionProvider ||
         provider_type == onnxruntime::kOpenVINOExecutionProvider ||
         provider_type == onnxruntime::kNnapiExecutionProvider ||
         provider_type == onnxruntime::kAclExecutionProvider ||
         provider_type == onnxruntime::kArmNNExecutionProvider ||
         provider_type == onnxruntime::kRknpuExecutionProvider ||
         provider_type == onnxruntime::kCoreMLExecutionProvider ||
         provider_type == onnxruntime::kSnpeExecutionProvider ||
         provider_type == onnxruntime::kXnnpackExecutionProvider ||
         provider_type == onnxruntime::utils::kInternalTestingExecutionProvider;
}

static common::Status AllocateHelper(const AllocatorPtr& allocator,
                                     const OrtValue& source_mlvalue,
                                     OrtValue& target_mlvalue) {
  if (!allocator) {
    return Status(common::ONNXRUNTIME, common::FAIL, "invalid allocator.");
  }

  if (source_mlvalue.IsTensor()) {
    const Tensor& source_tensor = source_mlvalue.Get<Tensor>();
    Tensor::InitOrtValue(source_tensor.DataType(),
                         source_tensor.Shape(),
                         allocator, target_mlvalue);
  } else if (source_mlvalue.IsSparseTensor()) {
#if !defined(DISABLE_SPARSE_TENSORS)
    const SparseTensor& source_tensor = source_mlvalue.Get<SparseTensor>();
    SparseTensor::InitOrtValue(source_tensor.DataType(), source_tensor.DenseShape(), allocator, target_mlvalue);
#endif
  } else if (source_mlvalue.IsTensorSequence()) {
    const TensorSeq& source_tensor_seq = source_mlvalue.Get<TensorSeq>();
    auto target_tensor_seq = std::make_unique<TensorSeq>(source_tensor_seq.DataType());
    std::vector<Tensor> tensors;
    for (auto iter = source_tensor_seq.begin(); iter != source_tensor_seq.end(); ++iter) {
      tensors.emplace_back(iter->DataType(), onnxruntime::TensorShape(iter->Shape()), allocator);
    }
    target_tensor_seq->SetElements(std::move(tensors));
    auto ml_tensor_seq = DataTypeImpl::GetType<TensorSeq>();
    target_mlvalue.Init(target_tensor_seq.release(), ml_tensor_seq, ml_tensor_seq->GetDeleteFunc());
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported OrtValue type.");
  }
  return Status::OK();
}

const std::string& GetNodeInputProviderType(const SessionState::NodeInfo& info) {
  // the input index will be std::numeric_limits<size_t>::max() if it's an implicit input to a control flow node.
  // the input will be processed fully when executing the subgraph that consumes the implicit input.
  bool implicit_input = info.index == std::numeric_limits<size_t>::max();

  // node may declare input_mem_type to be on CPU explicitly
  // skip implicit inputs as they don't have a valid 'index' value
  bool node_input_on_cpu = !implicit_input && info.kci && info.kci->kernel_def->IsInputOnCpu(info.index);

  // need a std::string that doesn't go away for kCpuExecutionProvider so we can return a reference.
  static const std::string cpu_execution_provider{onnxruntime::kCpuExecutionProvider};

  auto& required_provider_type = node_input_on_cpu ? cpu_execution_provider
                                                   : info.p_node->GetExecutionProviderType();

  return required_provider_type;
}

// Copy MLValue. Uses DataTransferManager for device copy if necessary. If copy_tensor_pairs/copy_sparse_pairs is provided,
// src/dst pairs that need a device copy are added to copy_pairs so copying can be batches by the DataTransferManager
// implementation for performance reasons.
static Status BatchOrCopyMLValue(const SessionState& session_state,
                                 const MLValueCopyInfo& copy_info,
                                 const OrtValue& source_mlvalue,
                                 OrtValue& target_mlvalue,
#if !defined(DISABLE_SPARSE_TENSORS)
                                 std::vector<IDataTransfer::SrcDstPair>* copy_tensor_pairs = nullptr,
                                 std::vector<IDataTransfer::SparseSrcDstPair>* copy_sparse_pairs = nullptr)
#else
                                 std::vector<IDataTransfer::SrcDstPair>* copy_tensor_pairs = nullptr)
#endif
{
  // same device so direct copy
  if (copy_info.source_device == copy_info.target_device) {
    target_mlvalue = source_mlvalue;
    return Status::OK();
  }

  auto allocator = session_state.GetAllocator(copy_info.target_device);
  if (!target_mlvalue.IsAllocated()) {
    ORT_ENFORCE(allocator != nullptr, "Failed to find allocator for device ", copy_info.target_device.ToString());
    ORT_RETURN_IF_ERROR(utils::AllocateHelper(allocator, source_mlvalue, target_mlvalue));
  }

  if (source_mlvalue.IsTensor()) {
    const auto& source_tensor = source_mlvalue.Get<Tensor>();
    Tensor* p_output_tensor = target_mlvalue.GetMutable<Tensor>();

    if (copy_tensor_pairs != nullptr) {
      copy_tensor_pairs->push_back({source_tensor, *p_output_tensor, 0});
    } else {
      ORT_RETURN_IF_ERROR(session_state.GetDataTransferMgr().CopyTensor(source_tensor, *p_output_tensor));
    }
  } else if (source_mlvalue.IsSparseTensor()) {
#if !defined(DISABLE_SPARSE_TENSORS)
    const auto& source_tensor = source_mlvalue.Get<SparseTensor>();
    SparseTensor* p_output_tensor = target_mlvalue.GetMutable<SparseTensor>();
    if (copy_sparse_pairs != nullptr) {
      copy_sparse_pairs->push_back({source_tensor, *p_output_tensor, 0});
    } else {
      ORT_RETURN_IF_ERROR(session_state.GetDataTransferMgr().CopySparseTensor(source_tensor, *p_output_tensor));
    }
#endif
  } else if (source_mlvalue.IsTensorSequence()) {
    const TensorSeq& source_tensor_seq = source_mlvalue.Get<TensorSeq>();
    TensorSeq& target_tensor_seq = const_cast<TensorSeq&>(target_mlvalue.Get<TensorSeq>());
    size_t size = 0;
    while ((size = target_tensor_seq.Size()) < source_tensor_seq.Size()) {
      if (0 == size) {
        target_tensor_seq.SetType(source_tensor_seq.DataType());
      }
      const Tensor& source_tensor = source_tensor_seq.Get(size);
      std::unique_ptr<Tensor> target_tensor = std::make_unique<Tensor>(source_tensor.DataType(), source_tensor.Shape(), allocator);
      target_tensor_seq.Add(std::move(*target_tensor));
    }
    auto source_iter = source_tensor_seq.begin();
    auto target_iter = target_tensor_seq.begin();
    while (source_iter != source_tensor_seq.end() &&
           target_iter != target_tensor_seq.end()) {
      if (copy_tensor_pairs != nullptr) {
        copy_tensor_pairs->push_back({*source_iter, const_cast<Tensor&>(*target_iter), 0});
      } else {
        ORT_RETURN_IF_ERROR(session_state.GetDataTransferMgr().CopyTensor(*source_iter, const_cast<Tensor&>(*target_iter)));
      }
      ++source_iter;
      ++target_iter;
    }  // while
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported OrtValue type to copy between device.");
  }

  return Status::OK();
}  // namespace utils

static bool HaveCpuExecutionProvidersOnly(const ExecutionProviders& execution_providers) {
  for (const auto& execution_provider : execution_providers) {
    if (!ProviderIsCpuBased(execution_provider->Type())) {
      return false;
    }
  }

  return true;
}

static const OrtMemoryInfo& FindMemoryInfoForValue(const OrtValueNameIdxMap& map,
                                                   const SequentialExecutionPlan& plan,
                                                   std::string_view name) {
  int idx = -1;
  auto status = map.GetIdx(name, idx);
  ORT_THROW_IF_ERROR(status);

  const auto& location = plan.GetLocation(idx);
  return location;
}

const OrtMemoryInfo& FindMemoryInfoForValue(const SessionState& session_state,
                                            std::string_view name) {
  const auto* exec_plan_ptr = session_state.GetExecutionPlan();
  ORT_ENFORCE(exec_plan_ptr);

  return FindMemoryInfoForValue(session_state.GetOrtValueNameIdxMap(), *exec_plan_ptr, name);
}

// get the target device info for the node consuming each input provided in the feeds.
// source_device info is not known until runtime
static common::Status CalculateStaticCopyInfoForFeed(const SessionState& session_state,
                                                     const std::string& input_name,
                                                     MLValueCopyInfo& copy_info) {
  InlinedVector<SessionState::NodeInfo> node_info_vec;
#ifdef ENABLE_TRAINING
  if (session_state.GetInputNodeInfo(input_name, node_info_vec) == Status::OK()) {
#else
  ORT_RETURN_IF_ERROR(session_state.GetInputNodeInfo(input_name, node_info_vec));
#endif
    const auto& node_info = node_info_vec.front();  // all consumers of a feed have the same device so first entry is fine

    if (node_info.p_node == nullptr) {
      // ignore dummy entry for an input that we didn't find a use of in the graph.
      return Status::OK();
    }

    copy_info.target_device = *node_info.device;

#ifdef ENABLE_TRAINING
  } else {
    // This input might be for an intermediate tensor for partial graph execution.
    const auto* exec_plan = session_state.GetExecutionPlan();
    const auto& name_to_id = session_state.GetOrtValueNameIdxMap();
    int index;
    ORT_RETURN_IF_ERROR(name_to_id.GetIdx(input_name, index));
    const auto& device = exec_plan->GetLocation(index).device;
    copy_info.target_device = device;
  }
#endif

  return Status::OK();
}

static common::Status CalculateStaticCopyInfoForFeeds(const SessionState& session_state,
                                                      gsl::span<const std::string> feed_names,
                                                      std::vector<MLValueCopyInfo>& copy_info) {
  for (size_t idx = 0, end = feed_names.size(); idx < end; ++idx) {
    ORT_RETURN_IF_ERROR(CalculateStaticCopyInfoForFeed(session_state, feed_names[idx], copy_info[idx]));
  }

  return Status::OK();
}

// get the source device info for the node producing each output that we will return in the fetches.
// target device info is not known until runtime.
static common::Status CalculateStaticCopyInfoForFetches(const SessionState& session_state,
                                                        gsl::span<const std::string> fetch_names,
                                                        std::vector<MLValueCopyInfo>& copy_info) {
  for (size_t idx = 0, end = fetch_names.size(); idx < end; ++idx) {
    const std::string& output_name = fetch_names[idx];

    const auto& info = FindMemoryInfoForValue(session_state, output_name);
    copy_info[idx].source_device = info.device;

    // If for some reason using just the device from the allocation plan isn't enough, the following
    // would use the NodeInfo from the node producing the output
    //
    // std::vector<SessionState::NodeInfo> node_info_vec;
    // auto status = session_state.GetOutputNodeInfo(output_name, node_info_vec);
    // if (status.IsOK()) {
    //  const auto& node_info = node_info_vec.front();  // only one entry as only one node can produce a given output
    //  copy_info[idx].source_device = *node_info.device;
    //} else {
    //  // edge case where an initializer directly provides output so no NodeInfo involved
    //  const auto& info = FindMemoryInfoForValue(session_state, output_name);
    //  copy_info[idx].source_device = info.device;
    //}
  }

  return Status::OK();
}

common::Status InitializeFeedFetchCopyInfo(const SessionState& session_state,
                                           FeedsFetchesManager& feeds_fetches_manager) {
  // if we only have CPU based EPs we can skip all the copy logic
  auto cpu_only = HaveCpuExecutionProvidersOnly(session_state.GetExecutionProviders());

  if (cpu_only) {
    feeds_fetches_manager.SetDeviceCopyChecks(DeviceCopyCheck::NoCopy, DeviceCopyCheck::NoCopy);
  } else {
    // setup all the static info about where the graph inputs and outputs are located
    auto info = feeds_fetches_manager.GetFeedsFetchesInfo();
    auto& feed_copy_info = feeds_fetches_manager.GetMutableFeedsDeviceCopyInfo();
    auto& fetch_copy_info = feeds_fetches_manager.GetMutableFetchesDeviceCopyInfo();
    ORT_RETURN_IF_ERROR(utils::CalculateStaticCopyInfoForFeeds(session_state, info.feed_names, feed_copy_info));
    ORT_RETURN_IF_ERROR(utils::CalculateStaticCopyInfoForFetches(session_state, info.output_names, fetch_copy_info));
  }

  return Status::OK();
}

// update the allocation_provider in the copy info based on the actual feeds
static bool FinalizeCopyInfoForFeeds(gsl::span<const OrtDevice> feed_locations,
                                     std::vector<MLValueCopyInfo>& copy_info) {
  ORT_ENFORCE(feed_locations.size() == copy_info.size());
  bool copy_needed = false;

  for (size_t i = 0, end = feed_locations.size(); i < end; ++i) {
    copy_info[i].source_device = feed_locations[i];

    if (copy_info[i].source_device != copy_info[i].target_device) {
      copy_needed = true;
    }
  }

  return copy_needed;
}

static bool FinalizeCopyInfoForFetches(gsl::span<const OrtMemoryInfo* const>& fetch_alloc_info,
                                       std::vector<MLValueCopyInfo>& copy_info) {
  ORT_ENFORCE(fetch_alloc_info.size() == copy_info.size());
  bool copy_needed = false;

  auto num_outputs = fetch_alloc_info.size();
  for (size_t i = 0; i < num_outputs; ++i) {
    const OrtMemoryInfo* alloc_info = fetch_alloc_info[i];

    if (alloc_info != nullptr) {
      copy_info[i].target_device = alloc_info->device;
    }

    if (copy_info[i].source_device != copy_info[i].target_device) {
      copy_needed = true;
    }
  }

  return copy_needed;
}

// Finalize the copy info using the OrtDevice and OrtMemoryInfo for the feeds and fetches
// This can be used by control flow nodes prior to the execution of the overall graph.
void FinalizeFeedFetchCopyInfo(FeedsFetchesManager& feeds_fetches_manager,
                               gsl::span<const OrtDevice> feed_locations,
                               gsl::span<const OrtMemoryInfo* const> fetch_alloc_info) {
  if (feeds_fetches_manager.GetDeviceCopyChecks().status == DeviceCopyCheck::NoCopy)
    return;

  bool need_copy = FinalizeCopyInfoForFeeds(feed_locations, feeds_fetches_manager.GetMutableFeedsDeviceCopyInfo());
  DeviceCopyCheck input_copy = need_copy ? DeviceCopyCheck::Copy : DeviceCopyCheck::NoCopy;

  need_copy = FinalizeCopyInfoForFetches(fetch_alloc_info, feeds_fetches_manager.GetMutableFetchesDeviceCopyInfo());
  DeviceCopyCheck output_copy = need_copy ? DeviceCopyCheck::Copy : DeviceCopyCheck::NoCopy;

  feeds_fetches_manager.SetDeviceCopyChecks(input_copy, output_copy);
}

// Finalize the copy info using the OrtValue instances for the feeds and fetches
static void FinalizeFeedFetchCopyInfo(FeedsFetchesManager& feeds_fetches_manager,
                                      gsl::span<const OrtValue> feeds,
                                      std::vector<OrtValue>& fetches) {
  if (feeds_fetches_manager.GetDeviceCopyChecks().status == DeviceCopyCheck::NoCopy)
    return;

  auto num_inputs = feeds.size();
  auto num_outputs = feeds_fetches_manager.GetFeedsFetchesInfo().output_names.size();

  std::vector<OrtDevice> feed_locations(num_inputs);
  std::vector<const OrtMemoryInfo*> fetch_alloc_info(num_outputs, nullptr);

  for (size_t i = 0; i < num_inputs; ++i) {
    const auto& feed = feeds[i];
    if (feed.IsTensor()) {
      feed_locations[i] = feed.Get<Tensor>().Location().device;
    } else if (feed.IsTensorSequence()) {
      const auto& tensor_seq = feed.Get<TensorSeq>();
      if (tensor_seq.Size() != std::size_t{0}) {
        feed_locations[i] = tensor_seq.Get(0).Location().device;
      }
    } else if (feed.IsSparseTensor()) {
#if !defined(DISABLE_SPARSE_TENSORS)
      feed_locations[i] = feed.Get<SparseTensor>().Location().device;
#endif
    }
  }

  // create default instances if needed
  fetches.resize(num_outputs);

  for (size_t i = 0; i < num_outputs; ++i) {
    const auto& fetch = fetches[i];
    if (fetch.IsAllocated()) {
      if (fetch.IsTensor()) {
        fetch_alloc_info[i] = &fetch.Get<Tensor>().Location();
      } else if (fetch.IsTensorSequence()) {
        const auto& tensor_seq = fetch.Get<TensorSeq>();
        if (tensor_seq.Size() != std::size_t{0}) {
          fetch_alloc_info[i] = &tensor_seq.Get(0).Location();
        }
      } else if (fetch.IsSparseTensor()) {
#if !defined(DISABLE_SPARSE_TENSORS)
        fetch_alloc_info[i] = &fetch.Get<SparseTensor>().Location();
#endif
      }
    }
  }

  FinalizeFeedFetchCopyInfo(feeds_fetches_manager, feed_locations, fetch_alloc_info);
}

static common::Status CopyInputsAcrossDevices(const SessionState& session_state,
                                              gsl::span<const OrtValue> orig_feeds,
                                              std::vector<OrtValue>& new_feeds,
                                              gsl::span<const MLValueCopyInfo> copy_info) {
  size_t num_feeds = orig_feeds.size();
  ORT_ENFORCE(copy_info.size() == num_feeds);

  new_feeds.resize(num_feeds);
  std::vector<IDataTransfer::SrcDstPair> batched_data_transfers;
#if !defined(DISABLE_SPARSE_TENSORS)
  std::vector<IDataTransfer::SparseSrcDstPair> batched_sparse_data_transfers;
#endif

  for (size_t idx = 0; idx < num_feeds; ++idx) {
#if !defined(DISABLE_SPARSE_TENSORS)
    ORT_RETURN_IF_ERROR(BatchOrCopyMLValue(session_state, copy_info[idx], orig_feeds[idx], new_feeds[idx],
                                           &batched_data_transfers, &batched_sparse_data_transfers));
#else
    ORT_RETURN_IF_ERROR(BatchOrCopyMLValue(session_state, copy_info[idx], orig_feeds[idx], new_feeds[idx],
                                           &batched_data_transfers));
#endif
  }

  if (!batched_data_transfers.empty()) {
    ORT_RETURN_IF_ERROR(session_state.GetDataTransferMgr().CopyTensors(batched_data_transfers));
  }

#if !defined(DISABLE_SPARSE_TENSORS)
  if (!batched_sparse_data_transfers.empty()) {
    ORT_RETURN_IF_ERROR(session_state.GetDataTransferMgr().CopySparseTensors(batched_sparse_data_transfers));
  }
#endif

  return Status::OK();
}

// public method to do a single copy. used by external partners
common::Status CopyOneInputAcrossDevices(const SessionState& session_state, const std::string& input_name,
                                         const OrtValue& orig_mlvalue, OrtValue& new_mlvalue) {
  if (!orig_mlvalue.IsTensor() && !orig_mlvalue.IsSparseTensor()) {
    new_mlvalue = orig_mlvalue;
    return Status::OK();
  }

  MLValueCopyInfo copy_info;
  // Sets copy_info.target_device.
  ORT_RETURN_IF_ERROR(CalculateStaticCopyInfoForFeed(session_state, input_name, copy_info));
#if !defined(DISABLE_SPARSE_TENSORS)
  copy_info.source_device = (orig_mlvalue.IsTensor())
                                ? orig_mlvalue.Get<Tensor>().Location().device
                                : orig_mlvalue.Get<SparseTensor>().Location().device;
#else
  copy_info.source_device = orig_mlvalue.Get<Tensor>().Location().device;
#endif

  // copy_info.target_device is not set leaving to be equal to CPU.
  return BatchOrCopyMLValue(session_state, copy_info, orig_mlvalue, new_mlvalue);
}

static common::Status CopyOutputsAcrossDevices(const SessionState& session_state,
                                               const std::vector<OrtValue>& fetches,
                                               std::vector<OrtValue>& user_fetches,
                                               const std::vector<MLValueCopyInfo>& copy_info) {
  auto num_outputs = fetches.size();
  user_fetches.resize(num_outputs);

  std::vector<IDataTransfer::SrcDstPair> batched_data_transfers;
#if !defined(DISABLE_SPARSE_TENSORS)
  std::vector<IDataTransfer::SparseSrcDstPair> batched_sparse_data_transfers;
#endif

  for (size_t idx = 0; idx < num_outputs; ++idx) {
#if !defined(DISABLE_SPARSE_TENSORS)
    ORT_RETURN_IF_ERROR(BatchOrCopyMLValue(session_state, copy_info[idx], fetches[idx], user_fetches[idx],
                                           &batched_data_transfers, &batched_sparse_data_transfers));
#else
    ORT_RETURN_IF_ERROR(BatchOrCopyMLValue(session_state, copy_info[idx], fetches[idx], user_fetches[idx],
                                           &batched_data_transfers));
#endif
  }

  if (!batched_data_transfers.empty()) {
    ORT_RETURN_IF_ERROR(session_state.GetDataTransferMgr().CopyTensors(batched_data_transfers));
  }

#if !defined(DISABLE_SPARSE_TENSORS)
  if (!batched_sparse_data_transfers.empty()) {
    ORT_RETURN_IF_ERROR(session_state.GetDataTransferMgr().CopySparseTensors(batched_sparse_data_transfers));
  }
#endif

  return Status::OK();
}

static common::Status ExecuteGraphImpl(const SessionState& session_state,
                                       const FeedsFetchesManager& feeds_fetches_manager,
                                       gsl::span<const OrtValue> feeds, std::vector<OrtValue>& fetches,
                                       const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                                       ExecutionMode execution_mode, const bool& terminate_flag,
                                       const logging::Logger& logger, const bool only_execute_path_to_fetches = false) {
  // avoid memory allocations
  std::optional<SequentialExecutor> seq_executor;
  std::optional<ParallelExecutor> par_executor;
  IExecutor* p_exec = nullptr;
  if (execution_mode == ExecutionMode::ORT_SEQUENTIAL) {
    seq_executor.emplace(terminate_flag, only_execute_path_to_fetches);
    p_exec = &seq_executor.value();
  } else if (execution_mode == ExecutionMode::ORT_PARALLEL) {
    auto* p_inter_op_thread_pool = session_state.GetInterOpThreadPool();
    if (!p_inter_op_thread_pool) {
      LOGS(logger, WARNING) << "Only one thread was configured for parallel execution. Hence will use sequential execution.";
      seq_executor.emplace(terminate_flag, only_execute_path_to_fetches);
      p_exec = &seq_executor.value();
    } else {
      par_executor.emplace(session_state, terminate_flag);
      p_exec = &par_executor.value();
    }
  }

  const auto& feeds_fetches_info = feeds_fetches_manager.GetFeedsFetchesInfo();
  const auto& device_copy_checks = feeds_fetches_manager.GetDeviceCopyChecks();

  // see if we can skip copies due to the types of execution providers available
  if (device_copy_checks.status == DeviceCopyCheck::NoCopy) {
    // no device copies are needed so simple execute
    ORT_RETURN_IF_ERROR(p_exec->Execute(session_state,
                                        feeds_fetches_info.feeds_mlvalue_idxs, feeds,
                                        feeds_fetches_info.fetches_mlvalue_idxs, fetches, fetch_allocators,
                                        logger));
  } else {
    auto feeds_to_use = feeds;
    std::vector<OrtValue>* p_fetches = &fetches;
    std::vector<OrtValue> device_feeds;
    std::vector<OrtValue> device_fetches;

    if (device_copy_checks.input_copy_needed == DeviceCopyCheck::Copy) {
      const auto& feed_copy_info = feeds_fetches_manager.GetFeedsDeviceCopyInfo();
      ORT_RETURN_IF_ERROR(CopyInputsAcrossDevices(session_state, feeds, device_feeds, feed_copy_info));
      feeds_to_use = device_feeds;
    }

    auto num_outputs = fetches.size();
    const auto& fetch_copy_info = feeds_fetches_manager.GetFetchesDeviceCopyInfo();

    if (device_copy_checks.output_copy_needed == DeviceCopyCheck::Copy) {
      // need intermediate fetches. use pre-allocated fetches where possible.
      device_fetches.reserve(num_outputs);

      for (size_t i = 0; i < num_outputs; ++i) {
        if (fetch_copy_info[i].source_device == fetch_copy_info[i].target_device && fetches[i].IsAllocated()) {
          device_fetches.push_back(fetches[i]);
        } else {
          // use temporary value
          device_fetches.push_back({});
        }
      }

      p_fetches = &device_fetches;
    }

    ORT_RETURN_IF_ERROR(p_exec->Execute(session_state,
                                        feeds_fetches_info.feeds_mlvalue_idxs, feeds_to_use,
                                        feeds_fetches_info.fetches_mlvalue_idxs, *p_fetches, fetch_allocators,
                                        logger));

    if (device_copy_checks.output_copy_needed == DeviceCopyCheck::Copy) {
      ORT_RETURN_IF_ERROR(CopyOutputsAcrossDevices(session_state, *p_fetches, fetches, fetch_copy_info));
    }
  }

  return Status::OK();
}

common::Status ExecuteGraph(const SessionState& session_state,
                            FeedsFetchesManager& feeds_fetches_manager,
                            gsl::span<const OrtValue> feeds, std::vector<OrtValue>& fetches,
                            ExecutionMode execution_mode, const bool& terminate_flag,
                            const logging::Logger& logger, bool only_execute_path_to_fetches) {
  ORT_RETURN_IF_ERROR(utils::InitializeFeedFetchCopyInfo(session_state, feeds_fetches_manager));

  // finalize the copy info using the provided feeds and fetches. will update device_copy_checks in the background
  FinalizeFeedFetchCopyInfo(feeds_fetches_manager, feeds, fetches);

  auto status = ExecuteGraphImpl(session_state, feeds_fetches_manager, feeds, fetches, {},
                                 execution_mode, terminate_flag, logger, only_execute_path_to_fetches);

  return status;
}

#ifdef ENABLE_TRAINING
common::Status ExecutePartialGraph(const SessionState& session_state, FeedsFetchesManager& feeds_fetches_manager,
                                   gsl::span<const OrtValue> feeds, std::vector<OrtValue>& fetches,
                                   const logging::Logger& logger, PartialGraphExecutionState& state,
                                   const OrtValueCachePtr& cache,
                                   int32_t partial_graph_index) {
  // finalize the copy info using the provided feeds and fetches. will update device_copy_checks in the background
  FinalizeFeedFetchCopyInfo(feeds_fetches_manager, feeds, fetches);
  PartialExecutor executor{state, cache, partial_graph_index};
  const auto& feeds_fetches_info = feeds_fetches_manager.GetFeedsFetchesInfo();
  const auto& device_copy_checks = feeds_fetches_manager.GetDeviceCopyChecks();

  // see if we can skip copies due to the types of execution providers available
  if (device_copy_checks.status == DeviceCopyCheck::NoCopy) {
    // no device copies are needed so simple execute
    ORT_RETURN_IF_ERROR(executor.Execute(session_state,
                                         feeds_fetches_info.feeds_mlvalue_idxs, feeds,
                                         feeds_fetches_info.fetches_mlvalue_idxs, fetches, {},
                                         logger));
  } else {
    auto p_feeds = feeds;
    std::vector<OrtValue>* p_fetches = &fetches;
    std::vector<OrtValue> device_feeds;
    std::vector<OrtValue> device_fetches;

    if (device_copy_checks.input_copy_needed == DeviceCopyCheck::Copy) {
      const auto& feed_copy_info = feeds_fetches_manager.GetFeedsDeviceCopyInfo();
      ORT_RETURN_IF_ERROR(CopyInputsAcrossDevices(session_state, feeds, device_feeds, feed_copy_info));
      p_feeds = device_feeds;
    }

    auto num_outputs = fetches.size();
    const auto& fetch_copy_info = feeds_fetches_manager.GetFetchesDeviceCopyInfo();

    if (device_copy_checks.output_copy_needed == DeviceCopyCheck::Copy) {
      // need intermediate fetches. use pre-allocated fetches where possible.
      device_fetches.reserve(num_outputs);

      for (size_t i = 0; i < num_outputs; ++i) {
        if (fetch_copy_info[i].source_device == fetch_copy_info[i].target_device && fetches[i].IsAllocated()) {
          device_fetches.push_back(fetches[i]);
        } else {
          // use temporary value
          device_fetches.push_back({});
        }
      }

      p_fetches = &device_fetches;
    }

    ORT_RETURN_IF_ERROR(executor.Execute(session_state,
                                         feeds_fetches_info.feeds_mlvalue_idxs, p_feeds,
                                         feeds_fetches_info.fetches_mlvalue_idxs, *p_fetches, {},
                                         logger));

    if (device_copy_checks.output_copy_needed == DeviceCopyCheck::Copy) {
      ORT_RETURN_IF_ERROR(CopyOutputsAcrossDevices(session_state, *p_fetches, fetches, fetch_copy_info));
    }
  }

  return Status::OK();
}
#endif

common::Status ExecuteSubgraph(const SessionState& session_state, const FeedsFetchesManager& feeds_fetches_manager,
                               gsl::span<const OrtValue> feeds, std::vector<OrtValue>& fetches,
                               const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                               ExecutionMode execution_mode, const bool& terminate_flag, const logging::Logger& logger) {
  auto status = ExecuteGraphImpl(session_state, feeds_fetches_manager, feeds, fetches, fetch_allocators,
                                 execution_mode, terminate_flag, logger);
  return status;
}

int32_t ONNXTensorElementDataTypeToProtoTensorType(ONNXTensorElementDataType onnx_enum) {
  switch (onnx_enum) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return onnx::TensorProto_DataType::TensorProto_DataType_FLOAT;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return onnx::TensorProto_DataType::TensorProto_DataType_DOUBLE;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return onnx::TensorProto_DataType::TensorProto_DataType_INT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return onnx::TensorProto_DataType::TensorProto_DataType_UINT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return onnx::TensorProto_DataType::TensorProto_DataType_INT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return onnx::TensorProto_DataType::TensorProto_DataType_UINT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return onnx::TensorProto_DataType::TensorProto_DataType_INT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return onnx::TensorProto_DataType::TensorProto_DataType_UINT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return onnx::TensorProto_DataType::TensorProto_DataType_INT64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return onnx::TensorProto_DataType::TensorProto_DataType_UINT64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
      return onnx::TensorProto_DataType::TensorProto_DataType_STRING;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return onnx::TensorProto_DataType::TensorProto_DataType_BOOL;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return onnx::TensorProto_DataType::TensorProto_DataType_FLOAT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      return onnx::TensorProto_DataType::TensorProto_DataType_BFLOAT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
      return onnx::TensorProto_DataType::TensorProto_DataType_COMPLEX64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
      return onnx::TensorProto_DataType::TensorProto_DataType_COMPLEX128;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
    default:
      assert(false);
      return onnx::TensorProto_DataType::TensorProto_DataType_UNDEFINED;
  }
}

#ifdef ENABLE_TRAINING
common::Status VerifyInputTensorsAllocatedContiguously(OpKernelContext* context) {
  const Tensor* prev_input = context->Input<Tensor>(0);
  for (int i = 1; i < context->InputCount(); i++) {
    const Tensor* curr_input = context->Input<Tensor>(i);

    ORT_ENFORCE(prev_input->Shape().Size() >= 0);

    const void* curr_address = curr_input->DataRaw();
    const void* prev_address = prev_input->DataRaw();
    const void* prev_end_address = reinterpret_cast<const char*>(prev_address) + prev_input->SizeInBytes();

    void* aligned_address = const_cast<void*>(prev_end_address);
    size_t dummy_space = kAllocAlignment * 2;
    std::align(kAllocAlignment, 1, aligned_address, dummy_space);

    if (!(curr_address == prev_end_address || curr_address == aligned_address)) {
      const std::string node = context->GetNodeName().empty() ? context->GetOpType() : context->GetNodeName();
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Contiguous memory checking failed on node ", node, ": ",
                             "input #", i - 1, " address is ", prev_address, " and #bytes = ", prev_input->SizeInBytes(),
                             ", input #", i, " address is ", curr_address);
    }

    prev_input = curr_input;
  }
  return Status::OK();
}
#endif

bool IsInputOnCpu(const Node& node, const KernelCreateInfo* p_kci, size_t index) {
  if (p_kci && p_kci->kernel_def->IsInputOnCpu(index)) {
    return true;
  }

#ifdef ENABLE_ATEN
  if (node.GetExecutionProviderType() == kCudaExecutionProvider && node.OpType() == "ATen" &&
      node.Domain() == kPytorchAtenDomain) {
    const auto& attrs = node.GetAttributes();
    ORT_ENFORCE(utils::HasString(attrs.at("operator")));
    std::string op_name = attrs.at("operator").s();
    std::string overload_name = "";
    if (attrs.find("overload_name") != attrs.end() && utils::HasString(attrs.at("overload_name"))) {
      overload_name = attrs.at("overload_name").s();
    }

    return !contrib::aten_ops::ATenOperatorExecutor::Instance().IsTensorArgument(op_name, overload_name, index);
  }
#else
  ORT_UNUSED_PARAMETER(node);
#endif

  return false;
}

}  // namespace utils
}  // namespace onnxruntime
