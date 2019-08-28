// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {
namespace utils {
void* DefaultAlloc(size_t size) {
  if (size <= 0) return nullptr;
  void* p;
  size_t alignment = MlasGetPreferredBufferAlignment();
#if _MSC_VER
  p = _aligned_malloc(size, alignment);
  if (p == nullptr) throw std::bad_alloc();
#elif defined(_LIBCPP_SGX_CONFIG)
  p = memalign(alignment, size);
  if (p == nullptr) throw std::bad_alloc();
#else
  int ret = posix_memalign(&p, alignment, size);
  if (ret != 0) throw std::bad_alloc();
#endif
  return p;
}

void DefaultFree(void* p) {
#if _MSC_VER
  _aligned_free(p);
#else
  free(p);
#endif
}

AllocatorPtr GetAllocator(const SessionState& session_state, const OrtAllocatorInfo& allocator_info) {
  return session_state.GetExecutionProviders().GetAllocator(allocator_info);
}

bool ProviderIsCpuBased(const std::string& provider_type) {
  return provider_type == onnxruntime::kCpuExecutionProvider ||
         provider_type == onnxruntime::kMklDnnExecutionProvider ||
         provider_type == onnxruntime::kNGraphExecutionProvider ||
         provider_type == onnxruntime::kNupharExecutionProvider ||
         provider_type == onnxruntime::kOpenVINOExecutionProvider ||
         provider_type == onnxruntime::kNnapiExecutionProvider;
}

common::Status AllocateHelper(const IExecutionProvider& execution_provider, const OrtDevice& device, const Tensor& fetched_tensor,
                              OrtValue& output_mlvalue) {
  auto allocator = execution_provider.GetAllocator(device.Id(), OrtMemTypeDefault);
  if (!allocator) {
    return Status(common::ONNXRUNTIME, common::FAIL, "invalid allocator");
  }

  std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(fetched_tensor.DataType(),
                                                              fetched_tensor.Shape(),
                                                              allocator);
  output_mlvalue.Init(p_tensor.release(),
                      DataTypeImpl::GetType<Tensor>(),
                      DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());

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

static Status CopyMLValue(const DataTransferManager& data_transfer_mgr,
                          const FeedsFetchesManager::MLValueCopyInfo& copy_info,
                          const OrtValue& source_mlvalue,
                          OrtValue& target_mlvalue) {
  if (copy_info.allocation_provider == nullptr) {
    target_mlvalue = source_mlvalue;
    return Status::OK();
  }

  auto& source_tensor = source_mlvalue.Get<Tensor>();
  if (!target_mlvalue.IsAllocated()) {
    ORT_RETURN_IF_ERROR(utils::AllocateHelper(*copy_info.allocation_provider, copy_info.target_device,
                                              source_tensor, target_mlvalue));
  }

  Tensor* p_output_tensor = target_mlvalue.GetMutable<Tensor>();

  ORT_RETURN_IF_ERROR(data_transfer_mgr.CopyTensor(source_tensor, *p_output_tensor));

  return Status::OK();
}

// TODO should we handle the case of one input name feeding 2 nodes placed on different devices?
common::Status CopyOneInputAcrossDevices(const SessionState& session_state, const std::string& input_name,
                                         const OrtValue& orig_mlvalue, OrtValue& new_mlvalue, bool& needed_copy,
                                         FeedsFetchesManager::MLValueCopyInfo& copy_info) {
  needed_copy = false;

  std::vector<SessionState::NodeInfo> node_info_vec;
  ORT_RETURN_IF_ERROR(session_state.GetInputNodeInfo(input_name, node_info_vec));

  auto& exec_providers = session_state.GetExecutionProviders();

  do {
    // currently we only support one device per input. see SessionState::AddInputNameToNodeInfoMapping for more
    // info on the logic to create the node_info_vec.
    // for (auto& node_info : node_info_vec) {
    auto& node_info = node_info_vec.front();
    if (node_info.p_node == nullptr) {
      // dummy entry for an input that we didn't find a use of in the graph.
      // use the input as is given we don't believe it's actually needed.
      new_mlvalue = orig_mlvalue;
      break;
    }

    if (!orig_mlvalue.IsTensor()) {
      // copying not supported for non-tensor types
      new_mlvalue = orig_mlvalue;
      break;
    }

    auto& required_device = *node_info.device;
    auto& input_tensor_device = orig_mlvalue.Get<Tensor>().Location().device;
    if (required_device == input_tensor_device) {
      // No copy needed for same device.
      new_mlvalue = orig_mlvalue;
      break;
    }

    auto& required_provider_type = GetNodeInputProviderType(node_info);
    auto* required_provider = exec_providers.Get(required_provider_type);
    copy_info.target_device = required_device;
    copy_info.allocation_provider = required_provider;

    ORT_RETURN_IF_ERROR(CopyMLValue(session_state.GetDataTransferMgr(), copy_info, orig_mlvalue, new_mlvalue));

    needed_copy = true;

  } while (false);

  return Status::OK();
}

common::Status CopyOneInputAcrossDevices(const SessionState& session_state, const std::string& input_name,
                                         const OrtValue& orig_mlvalue, OrtValue& new_mlvalue) {
  bool needed_copy;
  FeedsFetchesManager::MLValueCopyInfo ignored;
  return CopyOneInputAcrossDevices(session_state, input_name, orig_mlvalue, new_mlvalue, needed_copy, ignored);
}

// copies inputs across devices only if required and save copy_info
static common::Status CopyInputsAcrossDevices(const SessionState& session_state,
                                              const std::vector<std::string>& feed_names,
                                              const std::vector<OrtValue>& orig_feeds, std::vector<OrtValue>& new_feeds,
                                              bool& needed_copy,
                                              std::vector<FeedsFetchesManager::MLValueCopyInfo>* copy_info) {
  bool copied = false;
  size_t num_feeds = orig_feeds.size();
  ORT_ENFORCE(feed_names.size() == num_feeds);

  new_feeds.resize(num_feeds);
  if (copy_info) {
    copy_info->resize(num_feeds);
  }

  for (size_t idx = 0; idx < num_feeds; ++idx) {
    bool copied_this_input = false;
    FeedsFetchesManager::MLValueCopyInfo current_copy_info = {};  // init for each call
    ORT_RETURN_IF_ERROR(CopyOneInputAcrossDevices(session_state, feed_names[idx], orig_feeds[idx], new_feeds[idx],
                                                  copied_this_input, current_copy_info));

    if (copied_this_input) {
      copied = true;

      if (copy_info) {
        (*copy_info)[idx] = current_copy_info;
      }
    }
  }

  needed_copy = copied;

  return Status::OK();
}

// copies inputs across devices only if required using cached copy_info
static common::Status CachedCopyInputsAcrossDevices(
    const std::vector<OrtValue>& orig_feeds, std::vector<OrtValue>& new_feeds,
    const std::vector<FeedsFetchesManager::MLValueCopyInfo>& copy_info,
    const DataTransferManager& data_transfer_mgr) {
  size_t num_feeds = orig_feeds.size();
  ORT_ENFORCE(copy_info.size() == num_feeds);

  new_feeds.resize(num_feeds);

  for (size_t idx = 0; idx < num_feeds; ++idx) {
    ORT_RETURN_IF_ERROR(CopyMLValue(data_transfer_mgr, copy_info[idx], orig_feeds[idx], new_feeds[idx]));
  }

  return Status::OK();
}

// Setup fetches for execution. Use any provided fetches directly if the provider matches.
// If the provider doesn't match, we don't know what device the execution output may be on, so can't assume the output
// can be returned to the user directly.
static common::Status SetupFetchesForExecute(const SessionState& session_state,
                                             const std::vector<std::string>& output_names,
                                             std::vector<OrtValue>& fetches, std::vector<OrtValue>& new_fetches,
                                             std::vector<bool>* copy_to_new_fetches_cached_values) {
  ORT_ENFORCE(new_fetches.empty());
  auto num_outputs = output_names.size();
  new_fetches.resize(num_outputs);

  const auto& name_to_id = session_state.GetOrtValueNameIdxMap();
  const auto* exec_plan = session_state.GetExecutionPlan();
  // track which fetches can be copied to new_fetches and used directly in the execution.
  std::vector<bool> local_can_copy_flags(num_outputs, false);

  std::set<std::string> seen_outputs;
  auto p_graph = session_state.GetGraphViewer();
  ORT_ENFORCE(p_graph);

  auto contains = [](const std::vector<std::string>& output_names,
                     const std::string& name) {
    auto it = std::find(std::begin(output_names), std::end(output_names), name);
    if (it == output_names.end()) {
      return std::make_pair(false, size_t(0));
    }

    return std::pair<bool, size_t>(true, it - output_names.begin());
  };

  std::pair<bool, size_t> found;
  for (auto& node : p_graph->Nodes()) {
    if (seen_outputs.size() == num_outputs) {
      break;
    }

    for (auto* arg : node.OutputDefs()) {
      if (!arg->Exists() ||
          !(found = contains(output_names, arg->Name())).first) {
        continue;
      }

      seen_outputs.insert(arg->Name());
      size_t idx = found.second;
      const OrtValue& provided_mlvalue = fetches[idx];

      if (provided_mlvalue.IsAllocated()) {
        if (!provided_mlvalue.IsTensor()) {
          new_fetches[idx] = fetches[idx];
          local_can_copy_flags[idx] = true;
          continue;
        }

        int arg_index;
        ORT_RETURN_IF_ERROR(name_to_id.GetIdx(arg->Name(), arg_index));
        const auto& planned_device = exec_plan->GetLocation(arg_index).device;
        const auto& provided_tensor_device = provided_mlvalue.Get<Tensor>().Location().device;

        if (planned_device == provided_tensor_device) {
          new_fetches[idx] = fetches[idx];
          local_can_copy_flags[idx] = true;
          continue;
        }

        continue;
      }
    }
  }

  if (copy_to_new_fetches_cached_values) {
    *copy_to_new_fetches_cached_values = local_can_copy_flags;
  }

  return Status::OK();
}

static common::Status CachedSetupFetchesForExecute(std::vector<OrtValue>& fetches, std::vector<OrtValue>& new_fetches,
                                                   const std::vector<bool>& copy_to_new_fetches_cached_values) {
  auto num_outputs = fetches.size();
  ORT_ENFORCE(new_fetches.empty());
  ORT_ENFORCE(copy_to_new_fetches_cached_values.size() == num_outputs);

  new_fetches.resize(num_outputs);

  // use the cached values
  for (size_t i = 0; i < num_outputs; ++i) {
    if (copy_to_new_fetches_cached_values[i]) {
      new_fetches[i] = fetches[i];
    }
  }

  return Status::OK();
}

// copies outputs across devices only if required
static common::Status CopyOutputsAcrossDevices(const SessionState& session_state, const std::vector<OrtValue>& fetches,
                                               std::vector<OrtValue>& user_fetches, bool& needed_copy,
                                               std::vector<FeedsFetchesManager::MLValueCopyInfo>* copiers) {
  needed_copy = false;
  auto num_outputs = fetches.size();

  if (copiers) {
    // resize so we have default values and only need to update an entry if there's a device copy required.
    copiers->resize(num_outputs);
  }

  auto& execution_providers = session_state.GetExecutionProviders();

  // CPU execution provider is always registered so this is not null
  const auto* cpu_execution_provider = execution_providers.Get(onnxruntime::kCpuExecutionProvider);

  for (size_t idx = 0; idx < num_outputs; ++idx) {
    auto& fetched_mlvalue = fetches[idx];
    if (!fetched_mlvalue.IsTensor()) {
      user_fetches[idx] = fetched_mlvalue;
      continue;
    }

    const IExecutionProvider* p_output_provider = nullptr;
    auto target_device = OrtDevice();
    auto& output_mlvalue = user_fetches[idx];
    if (output_mlvalue.IsAllocated()) {
      Tensor* p_output_tensor = output_mlvalue.GetMutable<Tensor>();
      target_device = p_output_tensor->Location().device;
      p_output_provider = execution_providers.Get(p_output_tensor->Location());
    }
    auto fetch_result_device = fetched_mlvalue.Get<Tensor>().Location().device;
    if (target_device == fetch_result_device) {
      user_fetches[idx] = fetched_mlvalue;
      continue;
    }

    if (!p_output_provider) {
      p_output_provider = cpu_execution_provider;
    }

    needed_copy = true;
    FeedsFetchesManager::MLValueCopyInfo copy_info{target_device, p_output_provider};
    ORT_RETURN_IF_ERROR(CopyMLValue(session_state.GetDataTransferMgr(), copy_info, fetched_mlvalue, output_mlvalue));

    if (copiers) {
      (*copiers)[idx] = copy_info;
    }
  }

  return Status::OK();
}

static common::Status CachedCopyOutputsAcrossDevices(
    const std::vector<OrtValue>& fetches, std::vector<OrtValue>& user_fetches,
    const std::vector<FeedsFetchesManager::MLValueCopyInfo>& copy_info,
    const DataTransferManager& data_transfer_mgr) {
  auto num_outputs = fetches.size();

  // internal logic error if these are mismatched
  ORT_ENFORCE(num_outputs == copy_info.size());

  // used the cached copy logic if available
  for (size_t idx = 0; idx < num_outputs; ++idx) {
    ORT_RETURN_IF_ERROR(CopyMLValue(data_transfer_mgr, copy_info[idx], fetches[idx], user_fetches[idx]));
  }

  return Status::OK();
}

static DeviceCopyCheck CheckExecutionProviders(const ExecutionProviders& execution_providers) {
  for (const auto& execution_provider : execution_providers) {
    if (!ProviderIsCpuBased(execution_provider->Type())) {
      return DeviceCopyCheck::Unknown;
    }
  }

  return DeviceCopyCheck::NoCopy;
}

// execute graph with cached info from FeedsFetchesManager.
common::Status ExecuteGraphWithCachedInfo(
    const SessionState& session_state, const FeedsFetchesManager& feeds_fetches_manager,
    const std::vector<OrtValue>& feeds, std::vector<OrtValue>& fetches,
    const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators, bool sequential_execution,
    const bool& terminate_flag, const logging::Logger& logger) {
  const auto& feeds_fetches_info = feeds_fetches_manager.GetFeedsFetchesInfo();
  auto device_copy_checks = feeds_fetches_manager.GetDeviceCopyChecks();

  std::unique_ptr<IExecutor> p_exec;
  if (sequential_execution) {
    p_exec = std::unique_ptr<IExecutor>(new SequentialExecutor(terminate_flag));
  } else {
    p_exec = std::unique_ptr<IExecutor>(new ParallelExecutor(session_state, terminate_flag));
  }

  if (device_copy_checks.status == DeviceCopyCheck::NoCopy) {
    // no device copies are needed so simple execute
    ORT_RETURN_IF_ERROR(p_exec->Execute(session_state,
                                        feeds_fetches_info.feeds_mlvalue_idxs, feeds,
                                        feeds_fetches_info.fetches_mlvalue_idxs, fetches, fetch_allocators, logger));
  } else {
    const std::vector<OrtValue>* p_feeds = &feeds;
    std::vector<OrtValue>* p_fetches = &fetches;
    std::vector<OrtValue> device_feeds;
    std::vector<OrtValue> device_fetches;

    // Copy inputs
    if (device_copy_checks.input_copy_needed == DeviceCopyCheck::Copy) {
      ORT_RETURN_IF_ERROR(CachedCopyInputsAcrossDevices(feeds, device_feeds,
                                                        feeds_fetches_manager.GetFeedsDeviceCopiers(),
                                                        session_state.GetDataTransferMgr()));
      p_feeds = &device_feeds;
    }

    // setup fetches.
    if (fetches.empty()) {
      fetches.resize(feeds_fetches_info.output_names.size());
    }

    // if no output copy is needed, we can just use the fetches directly. otherwise we need to use a temporary set
    // and run CopyOutputsAcrossDevices.
    if (device_copy_checks.output_copy_needed == DeviceCopyCheck::Copy) {
      ORT_RETURN_IF_ERROR(CachedSetupFetchesForExecute(fetches, device_fetches,
                                                       feeds_fetches_manager.GetCanUseFetchDuringExecutionFlags()));
      p_fetches = &device_fetches;
    }

    ORT_RETURN_IF_ERROR(p_exec->Execute(session_state,
                                        feeds_fetches_info.feeds_mlvalue_idxs, *p_feeds,
                                        feeds_fetches_info.fetches_mlvalue_idxs, *p_fetches, fetch_allocators,
                                        logger));

    if (device_copy_checks.output_copy_needed == DeviceCopyCheck::Copy) {
      ORT_RETURN_IF_ERROR(CachedCopyOutputsAcrossDevices(*p_fetches, fetches,
                                                         feeds_fetches_manager.GetFetchesDeviceCopiers(),
                                                         session_state.GetDataTransferMgr()));
    }
  }

  return Status::OK();
}

// execute graph and update feeds_fetches_manager with cached copy info if cache_copy_info is true
common::Status ExecuteGraph(const SessionState& session_state, FeedsFetchesManager& feeds_fetches_manager,
                            const std::vector<OrtValue>& feeds, std::vector<OrtValue>& fetches,
                            const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                            bool sequential_execution, const bool& terminate_flag, const logging::Logger& logger,
                            bool cache_copy_info) {
  const auto& feeds_fetches_info = feeds_fetches_manager.GetFeedsFetchesInfo();
  auto device_copy_checks = feeds_fetches_manager.GetDeviceCopyChecks();

  ORT_ENFORCE(device_copy_checks.status == DeviceCopyCheck::Unknown);

  std::unique_ptr<IExecutor> p_exec;
  if (sequential_execution) {
    p_exec = std::unique_ptr<IExecutor>(new SequentialExecutor(terminate_flag));
  } else {
    p_exec = std::unique_ptr<IExecutor>(new ParallelExecutor(session_state, terminate_flag));
  }

  // see if we can skip copies due to the types of execution providers available
  if (CheckExecutionProviders(session_state.GetExecutionProviders()) == DeviceCopyCheck::NoCopy) {
    device_copy_checks.input_copy_needed = DeviceCopyCheck::NoCopy;
    device_copy_checks.output_copy_needed = DeviceCopyCheck::NoCopy;

    // no device copies are needed so simple execute
    ORT_RETURN_IF_ERROR(p_exec->Execute(session_state,
                                        feeds_fetches_info.feeds_mlvalue_idxs, feeds,
                                        feeds_fetches_info.fetches_mlvalue_idxs, fetches, fetch_allocators, logger));
  } else {
    bool copy_needed = false;

    const std::vector<OrtValue>* p_feeds = &feeds;
    std::vector<OrtValue>* p_fetches = &fetches;
    std::vector<OrtValue> device_feeds;
    std::vector<OrtValue> device_fetches;

    // Copy inputs
    auto* copiers = cache_copy_info ? &feeds_fetches_manager.GetMutableFeedsDeviceCopiers() : nullptr;
    ORT_RETURN_IF_ERROR(CopyInputsAcrossDevices(session_state,
                                                feeds_fetches_info.feed_names, feeds, device_feeds,
                                                copy_needed, copiers));

    if (copy_needed) {
      p_feeds = &device_feeds;
    }

    device_copy_checks.input_copy_needed = copy_needed ? DeviceCopyCheck::Copy
                                                       : DeviceCopyCheck::NoCopy;

    // setup fetches.
    if (fetches.empty()) {
      fetches.resize(feeds_fetches_info.output_names.size());
    }

    auto* use_provided_fetch_flags =
        cache_copy_info ? &feeds_fetches_manager.GetMutableCanUseFetchDuringExecutionFlags()
                        : nullptr;

    ORT_RETURN_IF_ERROR(SetupFetchesForExecute(session_state, feeds_fetches_info.output_names,
                                               fetches, device_fetches,
                                               use_provided_fetch_flags));
    p_fetches = &device_fetches;

    ORT_RETURN_IF_ERROR(p_exec->Execute(session_state,
                                        feeds_fetches_info.feeds_mlvalue_idxs, *p_feeds,
                                        feeds_fetches_info.fetches_mlvalue_idxs, *p_fetches, fetch_allocators,
                                        logger));

    copiers = cache_copy_info ? &feeds_fetches_manager.GetMutableFetchesDeviceCopiers() : nullptr;
    ORT_RETURN_IF_ERROR(CopyOutputsAcrossDevices(session_state, *p_fetches, fetches, copy_needed, copiers));

    device_copy_checks.output_copy_needed = copy_needed ? DeviceCopyCheck::Copy : DeviceCopyCheck::NoCopy;
  }

  // save the result of all the checks and use cached info next time
  if (cache_copy_info) {
    feeds_fetches_manager.SetDeviceCopyChecks(device_copy_checks);
  }

  return Status::OK();
}

#if defined(DEBUG_NODE_INPUTS_OUTPUTS)
std::ostream& operator<<(std::ostream& out, const BFloat16& value) {
  return out << value.ToFloat();
}

std::ostream& operator<<(std::ostream& out, const MLFloat16& value) {
  return out << value.val;
}

template <typename T>
static void DumpTensor(const Tensor& tensor, const TensorShape& shape) {
  auto num_items = shape.Size();

  if (num_items == 0) {
    std::cout << "no data";
    return;
  }

  size_t num_dims = shape.NumDimensions();
  size_t num_rows = 1;
  if (num_dims > 1) {
    num_rows = static_cast<size_t>(shape[0]);
  }

  size_t row_size = num_items / num_rows;

  auto data = tensor.DataAsSpan<T>();

  auto print_val = [](const T& value) {
    if (std::is_floating_point_v<T>)
      std::cout << std::setprecision(8) << value;
    else
      std::cout << value;
  };

  for (int row = 0; row < num_rows; ++row) {
    print_val(data[row * row_size]);
    for (int i = 1; i < row_size; ++i) {
      std::cout << ", ";
      print_val(data[row * row_size + i]);
    }
    std::cout << "\n";
  }

  std::cout << std::endl;
}

void DumpNodeInputs(const OpKernelContext& context, const Node& node) {
  std::cout << "-----------\n";
  std::cout << node.OpType() << " node: " << node.Name() << "\n";

  const auto& input_defs = node.InputDefs();

  for (auto i = 0, end = context.InputCount(); i < end; ++i) {
    if (input_defs[i]->Exists()) {
      std::cout << "Input " << i << " Name: " << input_defs[i]->Name();

      const auto* type = context.InputType(i);

      if (type) {
        if (type->IsTensorType()) {
          const auto& tensor = *context.Input<Tensor>(i);
          const auto& shape = tensor.Shape();

          std::cout << " Shape: " << shape << "\n";
        } else {
          std::cout << " is non-tensor type.\n";
        }
      } else {
        // should never happen...
        std::cout << " was missing data type\n";
      }
    } else {
      std::cout << "Input " << i << " is optional and was not provided.\n";
    }
  }
}

void DumpNodeOutputs(OpKernelContext& context, const Node& node, const SessionState& session_state) {
  std::cout << "-----------\n";
  const auto& output_defs = node.OutputDefs();

  const auto& execution_providers = session_state.GetExecutionProviders();
  const auto* cpu_execution_provider = execution_providers.Get(onnxruntime::kCpuExecutionProvider);

  for (auto i = 0, end = context.OutputCount(); i < end; ++i) {
    if (output_defs[i]->Exists()) {
      std::cout << "Output " << i << " Name: " << output_defs[i]->Name();

      const auto* type = context.OutputType(i);

      if (type) {
        if (type->IsTensorType()) {
          const auto& tensor = *context.Output<Tensor>(i);
          const auto data_type = tensor.DataType();
          const auto& shape = tensor.Shape();

          std::cout << " Shape: " << shape << "\n";

          // check tensor is on CPU before dumping it
          auto& tensor_location = tensor.Location();
          auto* provider = execution_providers.Get(tensor_location);
          if (!provider) {
            provider = cpu_execution_provider;
          }

          if (provider == cpu_execution_provider || tensor_location.mem_type == OrtMemTypeCPUOutput) {
            DispatchOnTensorType(data_type, DumpTensor, tensor, shape);
          } else {
            std::cout << " is not on CPU. Provider=" << provider->Type() << "\n";
          }
        } else {
          std::cout << " is non-tensor type.\n";
        }
      } else {
        // should never happen...
        std::cout << "missing data type\n";
      }
    } else {
      std::cout << "Output " << i << " is optional and was not produced.\n";
    }

    std::cout << std::endl;
  }
}
#endif

}  // namespace utils
}  // namespace onnxruntime
