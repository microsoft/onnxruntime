// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/spin_pause.h"
#include "core/framework/parallel_execution_plan.h"
#include "core/framework/session_state.h"
#include "core/framework/execution_frame.h"
#include "core/framework/utils.h"
#include "core/framework/mldata_type_utils.h"
#include "core/graph/constants.h"
#include "core/common/logging/macros.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/bfc_arena.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/execution_context.h"

#ifdef USE_CUDA
#include <nvtx3/nvToolsExtCuda.h>
#endif

#include <vector>

namespace onnxruntime {

struct Barrier {
  std::atomic_bool set_{false};
  void set() {
    set_.store(true);
  }
  void wait() {
    while (!set_.load()) {
      onnxruntime::concurrency::SpinPause();
    }
  }
};

using NotificationIndex = size_t;

struct ExecutionPlanContext;
using CommandFn = std::function<Status(void*, bool&)>;

// a logic stream to execute command.
// each command in the logic stream will be executed in FIFO
// a logic stream will be binded to multiple device stream, as the command in the same logic stream may be executed on different EPs.
// i.e., if we set concurrency level to 1, the single logic stream will be equal to our sequential execution plan, which has both cpu and gpu kernels
struct LogicStream {
  std::vector<CommandFn> commands_;
  const IExecutionProvider* ep_ = nullptr;
  void RunSince(ExecutionContext& ctx, size_t since);
  void RunSince(ExecutionPlanContext& ctx, size_t since);
  ~LogicStream() {}
};

void LogicStream::RunSince(ExecutionContext& ctx, size_t since) {
  if (!ctx.TaskStatus().IsOK()) {
    // already in bad status, terminate it
    ctx.CompleteTask();
    return;
  }
  while (since < commands_.size()) {
    if (ctx.TerminateFlag()) {
      ctx.SetStatus(ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Exiting due to terminate flag being set to true."));
      return;
    }
    bool continue_flag = true;
    Status status;
    ORT_TRY {
      status = commands_[since](&ctx, continue_flag);
    }
    ORT_CATCH(const std::exception& ex) {
      ORT_HANDLE_EXCEPTION([&]() {
        status = ORT_MAKE_STATUS(ONNXRUNTIME, RUNTIME_EXCEPTION, ex.what());
      });
    }
    if (!status.IsOK()) {
      //terminate it
      ctx.SetStatus(status);
      ctx.CompleteTask();
      return;
    }
    if (!continue_flag) {
      //break but not terminate
      return;
    }
    since++;
  }
  ctx.CompleteTask();
  return;
}

struct ParallelExecutionPlanImpl {
  ParallelExecutionPlanImpl(const SessionState& session_state,
                            const ProviderStreamMap& provider_stream_map,
                            const OpStreamMap& op_stream_map);
  ~ParallelExecutionPlanImpl();

  common::Status BindToDeviceStream(Stream* parent_stream, DeviceStreamColloection& device_stream_map) const;

  common::Status Execute(const SessionState& session_state,
                         const std::vector<int>& feed_mlvalue_idxs,
                         const std::vector<OrtValue>& feeds,
                         const std::vector<int>& fetch_mlvalue_idxs,
                         std::vector<OrtValue>& fetches,
                         const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                         const logging::Logger& logger,
                         const DeviceStreamColloection& device_streams,
                         const bool& terminate_flag,
                         const bool only_execute_path_to_fetches);

  const std::vector<int>& GetRefCounts() const { return value_ref_counts_; }

  size_t NumStreams() const { return num_logic_streams_; }

  void ScheduleDownstream(ExecutionContext& ctx, onnxruntime::NotificationIndex notification_index);

  const std::unordered_map<size_t, size_t>& GetValueToStreamMap() { return value_to_stream_map_; }

  std::vector<std::unique_ptr<LogicStream>> logic_streams_;
  const SessionState& session_state_;
  int num_logic_streams_{};

  // the stream where the notificaiton got created.
  std::vector<size_t> notification_owners_;
  std::unordered_map<NodeIndex, size_t> node_to_stream_map_;
  std::unordered_map<size_t, size_t> value_to_stream_map_;
  std::vector<int> value_ref_counts_;
  std::unordered_map<onnxruntime::NodeIndex, std::vector<onnxruntime::OrtValueIndex>> node_value_map_;
  std::unordered_map<onnxruntime::OrtValueIndex, onnxruntime::NodeIndex> value_node_map_;
  ProviderStreamMap provider_stream_map_;
  OpStreamMap op_stream_map_;
  std::vector<std::vector<std::string>> streams_log_;  // save up nodes per stream for logging

  int num_barriers_{};
  std::unordered_map<onnxruntime::NotificationIndex, std::vector<std::pair<int, int>>> downstream_map_;

  // dependence_graph_ keeps the dependencies combining model graph and logic streams
  // e.g. dependence_graph_[downstream_node] = [upstream_node_0, upstream_node_1, upstream_node_2 ...]
  // upstream_node_0 and upstream_node_1 are the immmediate upstream nodes of downstream_node
  // upstream_node_2 is the immediate nodes ahead of downstream_node is the same logic stream
  InlinedHashMap<onnxruntime::NodeIndex, InlinedHashSet<onnxruntime::NodeIndex>> dependence_graph_;
  std::unordered_map<onnxruntime::OrtValueIndex, std::unordered_set<onnxruntime::NodeIndex>> value_consumer_map_;
};

//todo: remove dependency on session_state

common::Status ParallelExecutionPlanImpl::BindToDeviceStream(Stream* parent_stream,
                                                             DeviceStreamColloection& device_stream_map) const {
  for (size_t i = 0; i < logic_streams_.size(); ++i) {
    auto& logic_stream = logic_streams_[i];
    if (logic_stream->commands_.size() > 0) {
      auto& stream_handle_registry = session_state_.GetStreamHandleRegistryInstance();
      auto create_stream_fn = stream_handle_registry.GetCreateStreamFn(logic_stream->ep_->Type());
      // TODO: in theory, we should make current subgraph's stream depends on parent stream.
      // but in current code structure, it causing issues with the resource sharing and stream
      // lifetime. it also may cause additional cost of stream sync for single stream case.
      // In first phase, let's just put all the subgraph execution on the parent stream.
      if (parent_stream) {
        // if current logic stream is not on the same EP instance as parent stream
        // and the EP instance does have async streams (not EP like CPU)
        // throw error as we don't have the code to setup the dependency at this moment.
        if (logic_stream->ep_ != parent_stream->provider && create_stream_fn) {
          ORT_THROW("Subgraph has nodes running on EP: ", logic_stream->ep_->Type(),
                    " while parent graph node running on EP: ", parent_stream->provider->Type(),
                    ", this is not supported yet.");
        }
        device_stream_map.SetDeviceStream(i, parent_stream);
      } else if (create_stream_fn) {
        auto device_stream = create_stream_fn(logic_stream->ep_);
        device_stream_map.SetDeviceStream(i, std::move(device_stream));
      } else {
        device_stream_map.SetDeviceStream(i, nullptr);
      }
    } else {
      device_stream_map.SetDeviceStream(i, nullptr);
    }
  }
  return Status::OK();
}

ParallelExecutionPlanImpl::ParallelExecutionPlanImpl(const SessionState& session_state,
                                                     const ProviderStreamMap& provider_stream_map,
                                                     const OpStreamMap& op_stream_map) : session_state_(session_state),
                                                                                         provider_stream_map_(provider_stream_map),
                                                                                         op_stream_map_(op_stream_map) {
  const auto& value_map = session_state_.GetOrtValueNameIdxMap();
  const auto& execution_providers = session_state_.GetExecutionProviders();
  const auto& kernel_create_info_map = session_state_.GetKernelCreateInfoMap();

  // instantiate logic streams

  class StreamRange {  //iterate between [from,to)
   public:
    StreamRange(int from, int to) : from_(from), to_(to){};
    int next() {
      int size = to_ - from_;
      ORT_ENFORCE(size > 0, "invalid stream range");
      int curr = from_ + iter_;
      iter_ = (iter_ + 1) % size;
      return curr;
    };
    StreamRange(const StreamRange&) = default;
    StreamRange(StreamRange&&) = default;
    StreamRange& operator=(const StreamRange&) = default;
    StreamRange& operator=(StreamRange&&) = default;

   private:
    int from_{};
    int to_{};
    int iter_{};
  };  //StreamRange

  int stream_idx = 0;

  std::map<std::string, std::unique_ptr<StreamRange>> stream_map;
  for (const auto& iter : provider_stream_map) {
    const auto& provider_name = iter.first;
    int num_streams = iter.second;
    int prev_stream_idx = stream_idx;
    for (int i = 0; i < num_streams; ++i) {
      logic_streams_.push_back(std::make_unique<LogicStream>());
      streams_log_.push_back(std::vector<std::string>{});  // todo - replace this with logger
      stream_idx++;
    }
    stream_map.insert({provider_name, std::make_unique<StreamRange>(prev_stream_idx, stream_idx)});
  }

  for (const auto& iter : op_stream_map_) {
    logic_streams_.push_back(std::make_unique<LogicStream>());
    streams_log_.push_back(std::vector<std::string>{});  // todo - replace this with logger
    for (const auto& op : iter) {
      stream_map.insert({op, std::make_unique<StreamRange>(stream_idx, stream_idx + 1)});
    }
    stream_idx++;
  }

  num_logic_streams_ = stream_idx;
  const auto& graph_viewer = session_state_.GetGraphViewer();

  //1. partition the nodes into streams
  std::unique_ptr<std::vector<NodeIndex>[]> nodes_in_stream { new std::vector<NodeIndex>[ num_logic_streams_ ] };
  std::unique_ptr<size_t[]> node_stream_map{new size_t[graph_viewer.MaxNodeIndex()]};
  for (auto node_index : graph_viewer.GetNodesInTopologicalOrder()) {
    const auto* node = graph_viewer.GetNode(node_index);
    int logic_stream_index = -1;
    if (stream_map.find(node->OpType()) != stream_map.end()) {
      logic_stream_index = stream_map[node->OpType()]->next();
    } else {
      onnxruntime::ProviderType exec_provider_name = node->GetExecutionProviderType();
      ORT_ENFORCE(stream_map.find(exec_provider_name) != stream_map.end());
      logic_stream_index = stream_map[exec_provider_name]->next();
    }
    ORT_ENFORCE(logic_stream_index > -1 && logic_stream_index < num_logic_streams_);
    nodes_in_stream[logic_stream_index].push_back(node_index);
    node_stream_map[node_index] = logic_stream_index;
    streams_log_[logic_stream_index].push_back(node->OpType());
  }

  //2. for each node, if any of its consumer partitioned to another stream, generate a notification
  size_t num_notifications = 0;
  std::unordered_map<NodeIndex, NotificationIndex> node_to_notification;
  for (auto i = 0; i < num_logic_streams_; ++i) {
    for (auto node_index : nodes_in_stream[i]) {
      auto* node = graph_viewer.GetNode(node_index);
      for (auto it = node->OutputNodesBegin(); it != node->OutputNodesEnd(); ++it) {
        if (std::find(nodes_in_stream[i].begin(), nodes_in_stream[i].end(), it->Index()) == nodes_in_stream[i].end()) {
          node_to_notification[node_index] = num_notifications++;
          break;
        }
      }
    }
  }
  //3. Check the nodes in each logical stream, build the map;
  for (auto i = 0; i < num_logic_streams_; ++i) {
    std::set<const IExecutionProvider*> providers;
    for (auto node_index : nodes_in_stream[i]) {
      node_to_stream_map_[node_index] = node_stream_map[node_index];
      auto* node = graph_viewer.GetNode(node_index);
      onnxruntime::ProviderType exec_provider_name = node->GetExecutionProviderType();
      const IExecutionProvider* ep = session_state.GetExecutionProviders().Get(exec_provider_name);
      if (logic_streams_[node_stream_map[node_index]]->ep_) {
        ORT_ENFORCE(logic_streams_[node_stream_map[node_index]]->ep_ == ep);
      } else {
        logic_streams_[node_stream_map[node_index]]->ep_ = ep;
      }
    }
  }
  //4. set notification owners
  notification_owners_.resize(num_notifications);
  for (auto node_index : graph_viewer.GetNodesInTopologicalOrder()) {
    auto it = node_to_notification.find(node_index);
    if (it != node_to_notification.end()) {
      // notification owned by the node who produced it.
      notification_owners_[it->second] = node_stream_map[node_index];
    }
  }
  //5. add commands to logic queue
  for (auto i = 0; i < num_logic_streams_; ++i) {
    for (auto j = 0; j < nodes_in_stream[i].size(); ++j) {
      auto node_index = nodes_in_stream[i][j];
      if (j > 0) {
        // add dependency for current logic stream
        dependence_graph_[node_index].insert(nodes_in_stream[i][j - 1]);
      }
      auto cur_stream_idx = node_to_stream_map_[node_index];
      // check if any producer is not in current stream, if yes, create a wait
      auto* node = graph_viewer.GetNode(node_index);
      for (auto it = node->InputNodesBegin(); it != node->InputNodesEnd(); ++it) {
        if (std::find(nodes_in_stream[i].begin(), nodes_in_stream[i].end(), it->Index()) == nodes_in_stream[i].end()) {
          // find the notificaiton id
          auto notfication_it = node_to_notification.find(it->Index());
          ORT_ENFORCE(notfication_it != node_to_notification.end());
          NotificationIndex notification_index = notfication_it->second;
          // push a barrier
          int barrier_id = num_barriers_++;
          downstream_map_[notification_index].push_back({i, static_cast<int>(logic_streams_[i]->commands_.size())});
          logic_streams_[i]->commands_.push_back([barrier_id](void* ctx, bool& continue_flag) {
            ExecutionContext* execution_context = reinterpret_cast<ExecutionContext*>(ctx);
            continue_flag = execution_context->DecCountDownBarrier(barrier_id);
            return Status::OK();
          });
          // push a wait command if has EP registered it.
          auto wait_handle = session_state.GetStreamHandleRegistryInstance().GetWaitHandle(
              logic_streams_[notification_owners_[notfication_it->second]]->ep_->Type(),
              node->GetExecutionProviderType());
          if (wait_handle) {
            const std::string& upstream_node_name = it->Name();
            logic_streams_[i]->commands_.push_back([wait_handle, cur_stream_idx, notification_index, i, node, upstream_node_name](void* ctx, bool& continue_flag) {
              ExecutionContext* execution_context = reinterpret_cast<ExecutionContext*>(ctx);
              wait_handle(*execution_context->GetDeviceStream(cur_stream_idx), *execution_context->GetNotification(notification_index));
              // update streams clock status
              if (execution_context->GetDeviceStream(cur_stream_idx)) {
                execution_context->GetDeviceStream(cur_stream_idx)->UpdateStreamClock(execution_context->GetNotification(notification_index)->stream_clock_);
              }
              LOGS(execution_context->GetLogger(), INFO) << "stream " << i << " wait on " << upstream_node_name << " for " << node->Name();
              continue_flag = true;
              return Status::OK();
            });
          }
        }
      }
      for (auto it = node->OutputNodesBegin(); it != node->OutputNodesEnd(); ++it) {
        // add dependency for model graph
        dependence_graph_[it->Index()].insert(node_index);
      }
      // push launch kernel command
      logic_streams_[i]->commands_.push_back([this, node, node_index, cur_stream_idx, i, &value_map](void* ctx, bool& continue_flag) {
        ExecutionContext* execution_context = reinterpret_cast<ExecutionContext*>(ctx);
        auto* p_kernel = execution_context->GetSessionState().GetKernel(node_index);
        auto* intra_tp = execution_context->GetSessionState().GetThreadPool();
        // TODO: set terminate flag from run_option
        OpKernelContextInternal kernel_ctx(execution_context->GetSessionState(), 
            *execution_context->GetExecutionFrame(), *p_kernel, execution_context->GetLogger(), 
            execution_context->TerminateFlag(), execution_context->GetDeviceStream(cur_stream_idx));
        // a temporary hack
        /*if (p_kernel->Info().GetKernelDef().OpName() == "If" || p_kernel->Info().GetKernelDef().OpName() == "Loop" || p_kernel->Info().GetKernelDef().OpName() == "Scan")
          if (kernel_ctx.GetComputeStream()) {
            kernel_ctx.GetComputeStream()->Flush();
          }*/
        if (p_kernel->IsAsync()) {
          ORT_THROW("Async Kernel Support is not implemented yet.");
        } else {
#ifdef USE_CUDA
          static std::atomic_int color = 0;
          color = (color + 10) / 1000;
          nvtxEventAttributes_t eventAttrib = {0};
          eventAttrib.version = NVTX_VERSION;
          eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
          eventAttrib.colorType = NVTX_COLOR_ARGB;
          eventAttrib.color = color;
          eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
          eventAttrib.message.ascii = p_kernel->Node().OpType().c_str();
          nvtxRangePushEx(&eventAttrib);
#endif
          ORT_RETURN_IF_ERROR(p_kernel->Compute(&kernel_ctx));

          execution_context->RecycleNodeInputs(node_index);
#ifdef USE_CUDA
          nvtxRangePop();
#endif
        }
        LOGS(execution_context->GetLogger(), INFO) << "stream " << i << " complete with " << node->Name();
        continue_flag = true;
        return Status::OK();
      });
      // check if any notification generated by this node, if yes, push a activate
      auto notification_it = node_to_notification.find(node_index);
      if (notification_it != node_to_notification.end()) {
        NotificationIndex notification_index = notification_it->second;
        logic_streams_[i]->commands_.push_back([notification_index, i, node](void* ctx, bool& continue_flag) {
          ExecutionContext* execution_context = reinterpret_cast<ExecutionContext*>(ctx);
          if (execution_context->GetNotification(notification_index)) {
            execution_context->GetNotification(notification_index)->ActivateAndUpdate();
          }
          LOGS(execution_context->GetLogger(), INFO) << "stream " << i << " send notification for " << node->Name();
          continue_flag = true;
          return Status::OK();
        });
        // notify downstreams
        logic_streams_[i]->commands_.push_back([this, notification_index](void* ctx, bool& continue_flag) {
          ExecutionContext* execution_context = reinterpret_cast<ExecutionContext*>(ctx);
          ScheduleDownstream(*execution_context, notification_index);
          continue_flag = true;
          return Status::OK();
        });
      }
    }
  }
  // 6. now prepare for release plan
  int num_ml_values = value_map.MaxIdx() + 1;
  std::unordered_set<int> node_outputs;
  value_ref_counts_.resize(num_ml_values);

  for (auto node_index : graph_viewer.GetNodesInTopologicalOrder()) {
    auto* node = graph_viewer.GetNode(node_index);
    const auto& output_defs = node->OutputDefs();
    for (int output_idx_local = 0; output_idx_local < output_defs.size(); ++output_idx_local) {
      const auto& node_output = output_defs[output_idx_local];
      if (!node_output->Exists()) continue;
      OrtValueIndex output_idx_global;
      ORT_THROW_IF_ERROR(value_map.GetIdx(node_output->Name(), output_idx_global));
      node_outputs.insert(output_idx_global);
      value_to_stream_map_[output_idx_global] = node_to_stream_map_[node_index];
      value_node_map_[output_idx_global] = node_index;
    }
  }

  for (auto node_index : graph_viewer.GetNodesInTopologicalOrder()) {
    auto* node = graph_viewer.GetNode(node_index);
    const auto& input_node_args = node->InputDefs();
    for (int input_index_local = 0; input_index_local < input_node_args.size(); ++input_index_local) {
      const auto* input_arg = input_node_args[input_index_local];
      //skip optional inputs
      if (input_arg->Exists()) {
        OrtValueIndex input_idx_global;
        ORT_THROW_IF_ERROR(value_map.GetIdx(input_arg->Name(), input_idx_global));
        value_consumer_map_[input_idx_global].insert(node_index);
        if (node_outputs.find(input_idx_global) != node_outputs.end()) {
          value_ref_counts_[input_idx_global]++;
          node_value_map_[node_index].push_back(input_idx_global);
        }
      }
    }
  }

  //TODO: move this to logger
  for (int i = 0; i < streams_log_.size(); ++i) {
    if (streams_log_[i].empty()) {
      std::cout << "stream " << i << ": <empty>" << std::endl;
    } else {
      std::stringstream ss;
      std::copy(streams_log_[i].begin(), streams_log_[i].end() - 1, std::ostream_iterator<std::string>(ss, ","));
      ss << streams_log_[i].back();
      std::cout << "stream " << i << ": " << ss.str() << std::endl;
    }
  }
}

void ParallelExecutionPlanImpl::ScheduleDownstream(ExecutionContext& ctx, onnxruntime::NotificationIndex notification_index) {
  auto* tp = session_state_.GetInterOpThreadPool();
  auto* ctx_ptr = &ctx;
  for (auto downstream : downstream_map_[notification_index]) {
    concurrency::ThreadPool::Schedule(tp, [this, ctx_ptr, downstream]() {
      logic_streams_[downstream.first]->RunSince(*ctx_ptr, downstream.second);
    });
  }
}

ParallelExecutionPlanImpl::~ParallelExecutionPlanImpl() {
}

common::Status ParallelExecutionPlanImpl::Execute(const SessionState& session_state, const std::vector<int>& feed_mlvalue_idxs,
                                                  const std::vector<OrtValue>& feeds, const std::vector<int>& fetch_mlvalue_idxs,
                                                  std::vector<OrtValue>& fetches,
                                                  const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                                                  const logging::Logger& logger,
                                                  const DeviceStreamColloection& device_streams,
                                                  const bool& terminate_flag,
                                                  const bool only_execute_path_to_fetches) {
  if (only_execute_path_to_fetches)
    ORT_THROW("NOT IMPLEMENTED YET.");
  int32_t valid_streams = 0;
  for (auto& stream : logic_streams_) {
    if (stream && stream->commands_.size() > 0)
      valid_streams++;
  }
  // prepare the execution context, notifications got initialized.
  ExecutionContext ctx(session_state,
                       valid_streams,
                       notification_owners_,
                       feed_mlvalue_idxs,
                       feeds,
                       fetch_mlvalue_idxs,
                       fetches,
                       fetch_allocators,
                       num_barriers_,
                       logger,
                       device_streams,
                       terminate_flag);

  auto* tp = session_state.GetInterOpThreadPool();

  for (int i = 0; i < num_logic_streams_; ++i) {
    if (!logic_streams_[i]->commands_.empty()) {
      concurrency::ThreadPool::Schedule(tp, [i, this, &ctx]() {
        logic_streams_[i]->RunSince(ctx, 0);
      });
    }
  }

  ctx.WaitAll();
  ORT_RETURN_IF_ERROR(ctx.TaskStatus());
  ORT_RETURN_IF_ERROR(ctx.GetExecutionFrame()->GetOutputs(fetches));
  return Status::OK();
}

ParallelExecutionPlan::ParallelExecutionPlan(const SessionState& session_state,
                                             const ProviderStreamMap& provider_stream_map,
                                             const OpStreamMap& op_stream_map) {
  impl_ = std::make_unique<ParallelExecutionPlanImpl>(session_state, provider_stream_map, op_stream_map);
}

ParallelExecutionPlan::~ParallelExecutionPlan() {
}

size_t ParallelExecutionPlan::NumStreams() const {
  return impl_->NumStreams();
}

common::Status ParallelExecutionPlan::Execute(const SessionState& session_state, const std::vector<int>& feed_mlvalue_idxs,
                                              const std::vector<OrtValue>& feeds, const std::vector<int>& fetch_mlvalue_idxs,
                                              std::vector<OrtValue>& fetches,
                                              const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                                              const logging::Logger& logger,
                                              const DeviceStreamColloection& device_streams,
                                              const bool& terminate_flag,
                                              const bool only_execute_path_to_fetches) {
  return impl_->Execute(session_state, feed_mlvalue_idxs, feeds, fetch_mlvalue_idxs, fetches, fetch_allocators, logger, device_streams, terminate_flag, only_execute_path_to_fetches);
}

const std::vector<AllocPlanPerValue>& ParallelExecutionPlan::GetAllocPlanPerValue() const {
  return this->allocation_plan;
}

const std::vector<int>& ParallelExecutionPlan::GetRefCounts() const {
  return impl_->value_ref_counts_;
}

const std::unordered_map<size_t, size_t>& ParallelExecutionPlan::GetValueToStreamMap() const {
  return impl_->GetValueToStreamMap();
}

common::Status ParallelExecutionPlan::BindToDeviceStream(Stream* parent_stream, DeviceStreamColloection& device_stream_map) const {
  return impl_->BindToDeviceStream(parent_stream, device_stream_map);
}

bool operator==(const onnx::TensorShapeProto& shape1, const onnx::TensorShapeProto& shape2) {
  namespace on = ONNX_NAMESPACE;
  int rank1 = shape1.dim_size();
  if (shape2.dim_size() != rank1) return false;
  for (int i = 0; i < rank1; i++) {
    const auto& val1 = shape1.dim(i);
    const auto& val2 = shape2.dim(i);
    if (utils::HasDimValue(val1) && utils::HasDimValue(val2) &&
        (val1.dim_value() == val2.dim_value()))
      continue;  // same known dimension
    if (utils::HasDimParam(val1) && utils::HasDimParam(val2)) {
      const auto& val1_param = val1.dim_param();
      if (val1_param == val2.dim_param() && !val1_param.empty())
        continue;  // same unknown dimension
    }
    return false;
  }
  return true;
}

std::unique_ptr<ReleasePlan> ParallelExecutionPlan::GenerateReleasePlan() const {
  auto release_plan = std::make_unique<ReleasePlan>();
  int num_values = impl_->session_state_.GetOrtValueNameIdxMap().MaxIdx() + 1;
  release_plan->value_ref_counts_.reset(new std::atomic_int[num_values]);
  return release_plan;
}

static bool SameShape(const ONNX_NAMESPACE::TensorShapeProto& shape1,
                      const ONNX_NAMESPACE::TensorShapeProto& shape2) {
  namespace on = ONNX_NAMESPACE;
  int rank1 = shape1.dim_size();
  if (shape2.dim_size() != rank1) return false;
  for (int i = 0; i < rank1; i++) {
    const auto& val1 = shape1.dim(i);
    const auto& val2 = shape2.dim(i);
    if (utils::HasDimValue(val1) && utils::HasDimValue(val2) &&
        (val1.dim_value() == val2.dim_value()))
      continue;  // same known dimension
    if (utils::HasDimParam(val1) && utils::HasDimParam(val2)) {
      const auto& val1_param = val1.dim_param();
      if (val1_param == val2.dim_param() && !val1_param.empty())
        continue;  // same unknown dimension
    }
    return false;
  }
  return true;
}

static size_t GetElementSize(const ONNX_NAMESPACE::DataType& tensor_type) {
  const ONNX_NAMESPACE::TypeProto& type_proto = ONNX_NAMESPACE::Utils::DataTypeUtils::ToTypeProto(tensor_type);
  MLDataType ml_data_type = DataTypeImpl::TypeFromProto(type_proto);
  const TensorTypeBase* tensor_type_base = ml_data_type->AsTensorType();
  ORT_ENFORCE(nullptr != tensor_type_base);
  MLDataType elt_type = tensor_type_base->GetElementType();
  return elt_type->Size();
}

static bool SameSize(const onnx::TensorShapeProto& shape1, const onnxruntime::NodeArg& arg1,
                     const onnx::TensorShapeProto& shape2, const onnxruntime::NodeArg& arg2) {
  const auto& ptype1 = arg1.Type();
  const auto& ptype2 = arg2.Type();
  auto type1_size = GetElementSize(ptype1);
  auto type2_size = GetElementSize(ptype2);
  bool is_type1_string = arg1.TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_STRING;
  bool is_type2_string = arg2.TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_STRING;
  return !(is_type1_string || is_type2_string) && (type1_size == type2_size) && SameShape(shape1, shape2);
}

bool SameSize(const ISequentialPlannerContext& context, const onnxruntime::NodeArg& arg1, const onnxruntime::NodeArg& arg2) {
  if ((!arg1.Exists()) || (!arg2.Exists())) return false;
  auto p_shape1 = context.GetShape(arg1);
  auto p_shape2 = context.GetShape(arg2);
  if ((nullptr == p_shape1) || (nullptr == p_shape2)) return false;
  return SameSize(*p_shape1, arg1, *p_shape2, arg2);
}

void ParallelExecutionPlan::GenerateReusePlan(const ISequentialPlannerContext& context) {
  InlinedHashMap<NodeIndex, int> dependents;
  for (const auto& it : impl_->dependence_graph_) {
    for (NodeIndex node_index : it.second) {
      dependents[node_index]++;
    }
  }
  std::deque<NodeIndex> que;
  for (const auto& it : impl_->dependence_graph_) {
    if (dependents[it.first] == 0) {
      que.push_back(it.first);
    }
  }

  // fetch_all_dependents will collect all dependent nodes for "node_index"
  std::function<std::set<NodeIndex>(NodeIndex)> fetch_all_dependents = [&](NodeIndex node_index) {
    std::set<NodeIndex> dependents;

    std::function<void(NodeIndex)> dfs = [&](NodeIndex curr) {
      if (dependents.find(curr) == dependents.end()) {
        dependents.insert(curr);
        for (NodeIndex dep : impl_->dependence_graph_[curr]) {
          dfs(dep);
        }
      }
    };

    dfs(node_index);
    return dependents;
  };

  // waiting_list keeps all values who want to reuse some upstream values' memory
  std::map<OrtMemoryInfo, std::map<size_t, typename std::map<const onnxruntime::NodeArg* const, std::set<NodeIndex>*>>> waiting_list;

  // for each node, dependents_map keeps all its dependent upstream nodes that are sure to be completed ahead
  std::map<NodeIndex, std::set<NodeIndex>> dependents_map;

  std::map<OrtValueIndex, std::set<OrtValueIndex>> input_output_map;

  std::set<OrtValueIndex> reused;

  const auto& graph_viewer = impl_->session_state_.GetGraphViewer();
  const auto& value_map = impl_->session_state_.GetOrtValueNameIdxMap();
  const auto& kernel_create_info_map = impl_->session_state_.GetKernelCreateInfoMap();
  const auto& allcation_plan = this->allocation_plan;

  std::function<void(NodeIndex)> TryReuseInput = [&](NodeIndex node_index) {
    auto* node = graph_viewer.GetNode(node_index);

    for (int output_arg_num = 0; output_arg_num < node->OutputDefs().size(); output_arg_num++) {
      auto p_output_arg = node->OutputDefs()[output_arg_num];
      OrtValueIndex output_idx_global{};

      if (!value_map.GetIdx(p_output_arg->Name(), output_idx_global).IsOK() ||
          allcation_plan[output_idx_global].alloc_kind != AllocKind::kAllocate) {
        continue;
      }

      auto kci_it = kernel_create_info_map.find(node_index);
      if (kci_it == kernel_create_info_map.end()) {
        continue;
      }

      const KernelCreateInfo& ci = *kci_it->second;
      if (ci.kernel_def == nullptr) {
        continue;
      }

      bool found_reusable = false;
      const auto& alias_map = ci.kernel_def->Alias();
      auto input_args = node->InputDefs();
      for (auto* input_arg : input_args) {
        OrtValueIndex input_idx_global{};
        if (value_map.GetIdx(input_arg->Name(), input_idx_global).IsOK()) {
          input_output_map[input_idx_global].insert(output_idx_global);
        }
      }

      for (auto& pair : alias_map) {
        if (pair.second == output_arg_num) {
          // we _must_ reuse this input to satisfy aliasing requirement: (e.g., for reshape)
          if ((0 <= pair.first) && (static_cast<size_t>(pair.first) < input_args.size())) {
            auto p_input_arg = input_args[pair.first];
            if (p_input_arg->Exists()) {
              OrtValueIndex reusable_input{};
              if (value_map.GetIdx(p_input_arg->Name(), reusable_input).IsOK() &&
                  allocation_plan[reusable_input].alloc_kind == AllocKind::kAllocate) {
                // LOGS(const_cast<SessionState&>(impl_->session_state_).Logger(), INFO) << p_input_arg->Name() << " reused by " << p_output_arg->Name() << " as input" << std::endl;
                std::cout << p_input_arg->Name() << " reused by " << p_output_arg->Name() << " as input" << std::endl;
                allocation_plan[output_idx_global].alloc_kind = AllocKind::kReuse;
                allocation_plan[output_idx_global].reused_buffer = reusable_input;
                impl_->value_consumer_map_[reusable_input].insert(impl_->value_consumer_map_[output_idx_global].begin(),
                                                                  impl_->value_consumer_map_[output_idx_global].end());
                reused.insert(reusable_input);
                found_reusable = true;
                break;
              }
            }
          }
        }
      }

      if (found_reusable) {
        continue;
      }

      const auto& variadic_alias_offsets = ci.kernel_def->VariadicAlias();
      if (variadic_alias_offsets.has_value()) {
        int input_offset = variadic_alias_offsets->first;
        int output_offset = variadic_alias_offsets->second;
        int alias_input_index = output_arg_num - output_offset + input_offset;

        if (alias_input_index >= 0 && static_cast<size_t>(alias_input_index) < input_args.size()) {
          auto p_input_arg = input_args[alias_input_index];

          if (p_input_arg->Exists()) {
            OrtValueIndex reusable_input{};
            if (value_map.GetIdx(p_input_arg->Name(), reusable_input).IsOK() &&
                allocation_plan[reusable_input].alloc_kind == AllocKind::kAllocate) {
              // LOGS(const_cast<SessionState&>(impl_->session_state_).Logger(), INFO) << p_input_arg->Name() << " reused by " << p_output_arg->Name() << " as input" << std::endl;
              std::cout << p_input_arg->Name() << " reused by " << p_output_arg->Name() << " as input" << std::endl;
              allocation_plan[output_idx_global].alloc_kind = AllocKind::kReuse;
              allocation_plan[output_idx_global].reused_buffer = reusable_input;
              impl_->value_consumer_map_[reusable_input].insert(impl_->value_consumer_map_[output_idx_global].begin(),
                                                                impl_->value_consumer_map_[output_idx_global].end());
              reused.insert(reusable_input);
              continue;
            }  //if
          }    //if
        }
      }

      const auto& inplace_map = ci.kernel_def->MayInplace();
      for (auto& pair : inplace_map) {
        if (pair.second == output_arg_num) {
          if ((0 <= pair.first) && (static_cast<size_t>(pair.first) < input_args.size())) {
            auto p_input_arg = input_args[pair.first];
            if (p_input_arg->Exists()) {
              OrtValueIndex input_arg_index{};
              if (value_map.GetIdx(p_input_arg->Name(), input_arg_index).IsOK() &&
                  allocation_plan[input_arg_index].alloc_kind == AllocKind::kAllocate) {
                if (impl_->value_consumer_map_[input_arg_index].size() == 1 && SameSize(context, *p_input_arg, *p_output_arg)) {
                  LOGS(const_cast<SessionState&>(impl_->session_state_).Logger(), INFO) << p_input_arg->Name() << " reused by " << p_output_arg->Name() << " as an input" << std::endl;
                  std::cout << p_input_arg->Name() << " reused by " << p_output_arg->Name() << " as an input" << std::endl;
                  allocation_plan[output_idx_global].alloc_kind = AllocKind::kReuse;
                  allocation_plan[output_idx_global].reused_buffer = input_arg_index;
                  impl_->value_consumer_map_[input_arg_index].insert(impl_->value_consumer_map_[output_idx_global].begin(),
                                                                     impl_->value_consumer_map_[output_idx_global].end());
                  reused.insert(input_arg_index);
                }
              }
            }
          }
        }
      }
    }
  };  //TryReuseInput

  // go over the outputs of "node_index" and try to reuse its memory
  std::function<void(NodeIndex)> TryReuseOutput = [&](NodeIndex node_index) {
    dependents_map[node_index] = fetch_all_dependents(node_index);
    auto* node = graph_viewer.GetNode(node_index);
    const auto& output_defs = node->OutputDefs();

    for (int output_idx_local = 0; output_idx_local < output_defs.size(); ++output_idx_local) {
      const auto& node_output = output_defs[output_idx_local];
      if (!node_output->Exists()) continue;
      OrtValueIndex output_idx_global{};

      if (value_map.GetIdx(node_output->Name(), output_idx_global).IsOK()) {
        if (reused.find(output_idx_global) != reused.end() ||
            allocation_plan[output_idx_global].alloc_kind != AllocKind::kAllocate) {
          continue;  // skip when it is already reused
        }

        const auto* shape = context.GetShape(*node_output);
        if (!shape) continue;
        size_t size_in_bytes = shape->ByteSizeLong();

        const auto& location = allocation_plan[output_idx_global].location;
        auto local_iter = waiting_list.find(location);
        if (local_iter == waiting_list.end()) {
          waiting_list[location][size_in_bytes][node_output] = &dependents_map[node_index];
          continue;
        }

        auto size_iter = local_iter->second.find(size_in_bytes);
        if (size_iter == local_iter->second.end()) {
          waiting_list[location][size_in_bytes][node_output] = &dependents_map[node_index];
          continue;
        }

        bool get_reused = false;
        for (auto node_iter = size_iter->second.begin(); node_iter != size_iter->second.end();) {
          const onnxruntime::NodeArg* const downstream_arg = node_iter->first;
          OrtValueIndex downstream_value{};

          if (!value_map.GetIdx(downstream_arg->Name(), downstream_value).IsOK()) {
            node_iter = next(node_iter);
            continue;
          }

          // skip if it is a pair of input and output
          if (input_output_map[output_idx_global].find(downstream_value) != input_output_map[output_idx_global].end()) {
            node_iter = next(node_iter);
            continue;
          }

          const auto* downstream_shape = context.GetShape(*downstream_arg);
          //if (!(*downstream_shape == *shape)) {
          //  node_iter = next(node_iter);
          //  continue;
          //}
          if (!SameSize(*downstream_shape, *downstream_arg, *shape, *node_output)) {
            node_iter = next(node_iter);
            continue;
          }

          auto* deps = node_iter->second;

          if (deps->find(node_index) == deps->end()) {
            node_iter = next(node_iter);
            continue;
          }

          bool all_covered = true;
          for (auto consumer : impl_->value_consumer_map_[output_idx_global]) {
            if (deps->find(consumer) == deps->end()) {
              all_covered = false;
              break;
            }
          }
          if (all_covered) {
            //LOGS(const_cast<SessionState&>(impl_->session_state_).Logger(), INFO) << node_output->Name() << " reused by " << downstream_arg->Name() << " as remote tensor" << std::endl;
            std::cout << node_output->Name() << " reused by " << downstream_arg->Name() << " as remote tensor" << std::endl;
            allocation_plan[downstream_value].alloc_kind = AllocKind::kReuse;
            allocation_plan[downstream_value].reused_buffer = output_idx_global;
            get_reused = true;
            // add new consumer for the value to be reused
            impl_->value_consumer_map_[output_idx_global].insert(impl_->value_node_map_[downstream_value]);
            impl_->value_consumer_map_[output_idx_global].insert(impl_->value_consumer_map_[downstream_value].begin(),
                                                                 impl_->value_consumer_map_[downstream_value].end());
            node_iter = size_iter->second.erase(node_iter);
            if (size_iter->second.empty()) {
              local_iter->second.erase(size_iter);
            }
            break;  // only resued once
          } else {
            // dependents not fully covered, cannot reuse, try next one in waiting_list
            node_iter = next(node_iter);
          }
        }  // for
        if (get_reused) {
          reused.insert(output_idx_global);
        } else {
          // if not getting reused, add to waiting
          waiting_list[location][size_in_bytes][node_output] = &dependents_map[node_index];
        }
      }
    }
  };  // TryReuseOutput

  // topological traverse of the dependency graph
  std::unordered_set<NodeIndex> visited;
  while (!que.empty()) {
    NodeIndex node_index = que.front();
    visited.insert(node_index);
    TryReuseInput(node_index);   // try reuse node's inputs as its outputs
    TryReuseOutput(node_index);  // try reuse node's outputs for downstream nodes
    que.pop_front();
    for (NodeIndex next_node_index : impl_->dependence_graph_[node_index]) {
      if (--dependents[next_node_index] == 0) {
        que.push_back(next_node_index);
      }
    }
  }
}

//////////////////////////////////////////////////// REFACTORED CLASSES /////////////////////////////////////////////////////////////

struct ExecutionPlanContext;
using LogicStreamOffset = std::pair<int, int>;

struct ExecutionPlanImpl {
  ReleasePlan release_plan;
  std::vector<AllocPlanPerValue> allocation_plan_;
  std::vector<std::unique_ptr<LogicStream>> logic_streams_;
  std::unordered_map<onnxruntime::NotificationIndex, std::vector<LogicStreamOffset>> downstream_map_;
  std::unordered_map<onnxruntime::OrtValueIndex, std::unordered_set<onnxruntime::NodeIndex>> value_consumer_map_;
  std::vector<size_t> notification_owners_;
  std::unordered_map<size_t, size_t> value_to_stream_map_;
  int num_barriers_{};

  void ScheduleDownstream(ExecutionPlanContext& ctx, onnxruntime::NotificationIndex notification_index);

  std::unique_ptr<ReleasePlan> GenerateReleasePlan() const {
    auto release_plan = std::make_unique<ReleasePlan>();
    release_plan->value_ref_counts_.reset(new std::atomic_int[allocation_plan_.size()]);
    return release_plan;
  }
};

ExecutionPlan::ExecutionPlan() {
  impl_ = std::make_unique<ExecutionPlanImpl>();
}

ExecutionPlan::~ExecutionPlan() {}

size_t ExecutionPlan::NumStreams() const {
    return impl_->logic_streams_.size();
}

common::Status ExecutionPlan::BindToDeviceStream(Stream* parent_stream,
                                                 DeviceStreamColloection& device_stream_map,
                                                 IStreamCommandHandleRegistry& stream_handle_registry) const {
  for (size_t i = 0; i < impl_->logic_streams_.size(); ++i) {
    auto& logic_stream = impl_->logic_streams_[i];
    if (logic_stream->commands_.size() > 0) {
      auto create_stream_fn = stream_handle_registry.GetCreateStreamFn(logic_stream->ep_->Type());
      // TODO: in theory, we should make current subgraph's stream depends on parent stream.
      // but in current code structure, it causing issues with the resource sharing and stream
      // lifetime. it also may cause additional cost of stream sync for single stream case.
      // In first phase, let's just put all the subgraph execution on the parent stream.
      if (parent_stream) {
        // if current logic stream is not on the same EP instance as parent stream
        // and the EP instance does have async streams (not EP like CPU)
        // throw error as we don't have the code to setup the dependency at this moment.
        if (logic_stream->ep_ != parent_stream->provider && create_stream_fn) {
          ORT_THROW("Subgraph has nodes running on EP: ", logic_stream->ep_->Type(),
                    " while parent graph node running on EP: ", parent_stream->provider->Type(),
                    ", this is not supported yet.");
        }
        device_stream_map.SetDeviceStream(i, parent_stream);
      } else if (create_stream_fn) {
        auto device_stream = create_stream_fn(logic_stream->ep_);
        device_stream_map.SetDeviceStream(i, std::move(device_stream));
      } else {
        device_stream_map.SetDeviceStream(i, nullptr);
      }
    } else {
      device_stream_map.SetDeviceStream(i, nullptr);
    }
  }
  return Status::OK();
}

const std::vector<AllocPlanPerValue>& ExecutionPlan::GetAllocationPlan() {
  return impl_->allocation_plan_;
}

const std::unordered_map<size_t, size_t>& ExecutionPlan::GetValueToStreamMap() const {
  return impl_->value_to_stream_map_;
}

static void CalculateTotalInputSizes(const OpKernelContextInternal* op_kernel_context,
                                     const onnxruntime::OpKernel* p_op_kernel,
                                     size_t& input_activation_sizes, size_t& input_parameter_sizes,
                                     const std::string& node_name, std::string& input_type_shape) {
  // Calculate total input sizes for this operation.
  std::stringstream ss;
  ss << "[";
  int added_type_shapes = 0;
  input_activation_sizes = 0;
  input_parameter_sizes = 0;
  ORT_UNUSED_PARAMETER(node_name);
  const int input_count = op_kernel_context->InputCount();
  for (auto i = 0; i < input_count; i++) {
    const OrtValue* p_input = op_kernel_context->GetInputMLValue(i);
    if (p_input != nullptr && p_input->IsTensor()) {
      const OpKernelInfo& op_kernel_info = p_op_kernel->Info();
      const Tensor* p_tensor = nullptr;
      bool is_param = op_kernel_info.TryGetConstantInput(i, &p_tensor);
      if (!is_param) {
        p_tensor = &(p_input->Get<Tensor>());
      }
      size_t tensor_size = p_tensor->SizeInBytes();

#if defined(TRACE_EXECUTION)
      const TensorShape& tensor_shape = p_tensor->Shape();
      size_t element_size = p_tensor->DataType()->Size();
      LOGS(logger, INFO) << node_name << " input[" << i << "]"
                         << " is_param=" << is_param
                         << " size=" << tensor_size
                         << " shape=" << tensor_shape.ToString()
                         << " element_size=" << element_size
                         << "\n";
#endif
      if (is_param) {
        input_parameter_sizes += tensor_size;
      } else {
        input_activation_sizes += tensor_size;
      }
      auto shape_str = p_tensor->Shape().ToString();
      ss << (added_type_shapes++ > 0 ? "," : "")
         << "{\"" << DataTypeImpl::ToString(p_tensor->DataType()) << "\":["
         << shape_str.substr(1, shape_str.size() - 2) << "]}";
    }
  }
  ss << "]";
  input_type_shape = ss.str();
}

struct ExecutionPlanContext {
  const SessionState& session_state_;
  const ExecutionPlanImpl& plan_;
  ExecutionFrame& frame_;
  const logging::Logger& logger_;
  std::vector<std::unique_ptr<synchronize::Notification>> notifications_;
  std::unique_ptr<ReleasePlan> release_plan_;
  std::vector<CountDownBarrier> count_down_barriers_;
  const DeviceStreamColloection& device_stream_map_;
  CountDownBarrier remain_tasks_;
  const bool& terminate_flag_;
  Status task_status_{Status::OK()};

  //members for profiling
  const bool is_profiler_enabled_;
  TimePoint tp;
  TimePoint sync_time_begin;
  TimePoint kernel_begin_time;
  size_t input_activation_sizes = 0;
  size_t input_parameter_sizes = 0;
  size_t total_output_sizes = 0;
  std::string input_type_shape{};
  std::string output_type_shape{};
  const Node* node_ {};

  ExecutionPlanContext(const SessionState& session_state,
                       const ExecutionPlan& exe_plan,
                       ExecutionFrame& frame,
                       const logging::Logger& logger,
                       const DeviceStreamColloection& device_streams_map,
                       const bool& terminate_flag) : session_state_(session_state),
                                                     plan_(*exe_plan.impl_),
                                                     frame_(frame),
                                                     logger_(logger),
                                                     count_down_barriers_(plan_.num_barriers_),
                                                     device_stream_map_(device_streams_map),
                                                     terminate_flag_(terminate_flag),
                                                     is_profiler_enabled_(session_state.Profiler().IsEnabled()) {
    int32_t valid_streams = 0;
    //1. bind logic stream to device stream;
    for (auto& logic_stream : plan_.logic_streams_) {
      if (logic_stream->commands_.size() > 0) {
        valid_streams++;
      }
    }

    auto& device_streams = device_stream_map_.GetStreams();
    for (auto i = 0; i < plan_.notification_owners_.size(); ++i) {
      auto& stream = device_streams[plan_.notification_owners_[i]];
      if (stream) {
        notifications_.push_back(std::move(stream->CreateNotification(/*TODO: calculate num of consumers*/ 0)));
      } else {
        notifications_.push_back(nullptr);
      }
    }

    // init barreris
    for (auto i = 0; i < plan_.num_barriers_; ++i) {
      count_down_barriers_[i].Set(2);
    }
    remain_tasks_.Set(valid_streams);
    release_plan_ = plan_.GenerateReleasePlan();
  }

  bool DecCountDownBarrier(size_t barrier_id) {
    return count_down_barriers_[barrier_id].Dec();
  }

  Stream* GetDeviceStream(size_t idx) {
    ORT_ENFORCE(idx < device_stream_map_.NumStreams());
    return device_stream_map_.GetStreams()[idx];
  }

  void CompleteTask() {
    remain_tasks_.Dec();
  }

  void WaitAll() {
    while (task_status_.IsOK() && remain_tasks_.Get()) {
      onnxruntime::concurrency::SpinPause();
    }
  }

  void SetStatus(Status& status) {
    // TODO: if multiple worker report non-ok status,
    // what is our strategy? currently we just keep
    // a random one. as long as it is not OK, the
    // execution will fail.
    if (task_status_.IsOK() && !status.IsOK())
      task_status_ = status;
  }

  ~ExecutionPlanContext() {}

  void RecycleNodeInputs(onnxruntime::NodeIndex /*node_index*/) {
    
  }

  onnxruntime::Status ExecuteKernel(onnxruntime::NodeIndex node_index, onnxruntime::HashValue stream_index) {
    auto* p_op_kernel = session_state_.GetKernel(node_index);
    auto* intra_tp = session_state_.GetThreadPool();
    OpKernelContextInternal kernel_context(session_state_, frame_, *p_op_kernel, logger_, terminate_flag_, GetDeviceStream(stream_index));

    if (is_profiler_enabled_) {
      node_ = session_state_.GetGraphViewer().GetNode(node_index);
      auto sync_time_begin = session_state_.Profiler().Start();
      LOGS(logger_, INFO) << "Computing kernel: " << node_->Name();
      session_state_.Profiler().EndTimeAndRecordEvent(profiling::NODE_EVENT,
                                                     node_->Name() + "_fence_before",
                                                     sync_time_begin,
                                                     {{"op_name", p_op_kernel->KernelDef().OpName()}});
      concurrency::ThreadPool::StartProfiling(session_state_.GetThreadPool());
      kernel_begin_time = session_state_.Profiler().Start();
      // Calculate total input sizes for this operation.
      CalculateTotalInputSizes(&kernel_context, p_op_kernel,
                               input_activation_sizes, input_parameter_sizes,
                               node_->Name(), input_type_shape);
    }

    ORT_RETURN_IF_ERROR(p_op_kernel->Compute(&kernel_context));

    if (is_profiler_enabled_) {
      session_state_.Profiler().EndTimeAndRecordEvent(profiling::NODE_EVENT,
                                                     node_->Name() + "_kernel_time",
                                                     kernel_begin_time,
                                                     // Log additional operation args / info.
                                                     {
                                                         {"op_name", p_op_kernel->KernelDef().OpName()},
                                                         {"provider", p_op_kernel->KernelDef().Provider()},
                                                         {"graph_index", std::to_string(p_op_kernel->Node().Index())},
                                                         {"exec_plan_index", std::to_string(node_index)},
                                                         {"activation_size", std::to_string(input_activation_sizes)},
                                                         {"parameter_size", std::to_string(input_parameter_sizes)},
                                                         {"output_size", std::to_string(total_output_sizes)},
                                                         {"input_type_shape", input_type_shape},
                                                         {"output_type_shape", output_type_shape},
                                                         {"thread_scheduling_stats", concurrency::ThreadPool::StopProfiling(session_state_.GetThreadPool())},
                                                     });
      auto sync_time_begin = session_state_.Profiler().Start();
      session_state_.Profiler().EndTimeAndRecordEvent(profiling::NODE_EVENT,
                                                     node_->Name() + "_fence_after",
                                                     sync_time_begin,
                                                     {{"op_name", p_op_kernel->KernelDef().OpName()}});
    }

    RecycleNodeInputs(node_index);
    return Status::OK();
  }
};

struct ExecutionPlannerImpl {
  const Node* parent_node_;
  const onnxruntime::GraphViewer& graph_viewer_;
  const std::vector<const NodeArg*>& outer_scope_node_args_;
  const ExecutionProviders& execution_providers_;
  const KernelCreateInfoMap& kernel_create_info_map_;
  const SubgraphsKernelCreateInfoMaps& subgraphs_kernel_create_info_maps_;
  const std::unordered_map<OrtValueName, OrtMemoryInfo>& outer_scope_node_arg_to_location_map_;
  const OrtValueNameIdxMap& ort_value_name_idx_map_;
  std::vector<int> value_ref_counts_;
  InlinedHashMap<onnxruntime::NodeIndex, InlinedHashSet<onnxruntime::NodeIndex>> dependence_graph_;
  std::unordered_map<onnxruntime::OrtValueIndex, onnxruntime::NodeIndex> value_node_map_;
  IStreamCommandHandleRegistry& stream_handle_registry_;
  const ProviderStreamMap& provider_stream_map_;
  const OpStreamMap& op_stream_map_;
  const ISequentialPlannerContext& context_;

  ExecutionPlannerImpl(const Node* parent_node,
                       const onnxruntime::GraphViewer& graph_viewer,
                       const std::vector<const NodeArg*>& outer_scope_node_args,
                       const ExecutionProviders& providers,
                       const KernelCreateInfoMap& kernel_create_info_map,
                       const SubgraphsKernelCreateInfoMaps& subgraphs_kernel_create_info_maps,
                       const std::unordered_map<OrtValueName, OrtMemoryInfo>& outer_scope_node_arg_to_location_map,
                       const OrtValueNameIdxMap& ort_value_name_idx_map,
                       IStreamCommandHandleRegistry& stream_handle_registry,
                       const ProviderStreamMap& provider_stream_map,
                       const OpStreamMap& op_stream_map,
                       const ISequentialPlannerContext& context) : parent_node_(parent_node),
                                                                   graph_viewer_(graph_viewer),
                                                                   outer_scope_node_args_(outer_scope_node_args),
                                                                   execution_providers_(providers),
                                                                   kernel_create_info_map_(kernel_create_info_map),
                                                                   subgraphs_kernel_create_info_maps_(subgraphs_kernel_create_info_maps),
                                                                   outer_scope_node_arg_to_location_map_(outer_scope_node_arg_to_location_map),
                                                                   ort_value_name_idx_map_(ort_value_name_idx_map),
                                                                   stream_handle_registry_(stream_handle_registry),
                                                                   provider_stream_map_(provider_stream_map),
                                                                   op_stream_map_(op_stream_map),
                                                                   context_(context) {}

  onnxruntime::Status CreatePlan(ExecutionPlan& exe_plan) {
    ExecutionPlanImpl& plan = *exe_plan.impl_;
    std::vector<std::vector<std::string>> streams_log;
    std::unordered_map<NodeIndex, size_t> node_to_stream_map;
    std::unordered_map<onnxruntime::NodeIndex, std::vector<onnxruntime::OrtValueIndex>> node_value_map;

    // instantiate logic streams
    class StreamRange {  //iterate between [from,to)
     public:
      StreamRange(int from, int to) : from_(from), to_(to){};
      int next() {
        int size = to_ - from_;
        ORT_ENFORCE(size > 0, "invalid stream range");
        int curr = from_ + iter_;
        iter_ = (iter_ + 1) % size;
        return curr;
      };
      StreamRange(const StreamRange&) = default;
      StreamRange(StreamRange&&) = default;
      StreamRange& operator=(const StreamRange&) = default;
      StreamRange& operator=(StreamRange&&) = default;

     private:
      int from_{};
      int to_{};
      int iter_{};
    };  //StreamRange

    int stream_idx = 0;

    std::map<std::string, std::unique_ptr<StreamRange>> stream_map;
    for (const auto& iter : provider_stream_map_) {
      const auto& provider_name = iter.first;
      int num_streams = iter.second;
      int prev_stream_idx = stream_idx;
      for (int i = 0; i < num_streams; ++i) {
        plan.logic_streams_.push_back(std::make_unique<LogicStream>());
        streams_log.push_back(std::vector<std::string>{});  // todo - replace this with logger
        stream_idx++;
      }
      stream_map.insert({provider_name, std::make_unique<StreamRange>(prev_stream_idx, stream_idx)});
    }

    for (const auto& iter : op_stream_map_) {
      plan.logic_streams_.push_back(std::make_unique<LogicStream>());
      streams_log.push_back(std::vector<std::string>{});  // todo - replace this with logger
      for (const auto& op : iter) {
        stream_map.insert({op, std::make_unique<StreamRange>(stream_idx, stream_idx + 1)});
      }
      stream_idx++;
    }

    int num_logic_streams = stream_idx;

    //1. partition the nodes into streams
    std::unique_ptr<std::vector<NodeIndex>[]> nodes_in_stream { new std::vector<NodeIndex>[ num_logic_streams ] };
    std::unique_ptr<size_t[]> node_stream_map{new size_t[graph_viewer_.MaxNodeIndex()]};
    for (auto node_index : graph_viewer_.GetNodesInTopologicalOrder()) {
      const auto* node = graph_viewer_.GetNode(node_index);
      int logic_stream_index = -1;
      if (stream_map.find(node->OpType()) != stream_map.end()) {
        logic_stream_index = stream_map[node->OpType()]->next();
      } else {
        onnxruntime::ProviderType exec_provider_name = node->GetExecutionProviderType();
        ORT_ENFORCE(stream_map.find(exec_provider_name) != stream_map.end());
        logic_stream_index = stream_map[exec_provider_name]->next();
      }
      ORT_ENFORCE(logic_stream_index > -1 && logic_stream_index < num_logic_streams);
      nodes_in_stream[logic_stream_index].push_back(node_index);
      node_stream_map[node_index] = logic_stream_index;
      streams_log[logic_stream_index].push_back(node->OpType());
    }

    //2. for each node, if any of its consumer partitioned to another stream, generate a notification
    size_t num_notifications = 0;
    std::unordered_map<NodeIndex, NotificationIndex> node_to_notification;
    for (auto i = 0; i < num_logic_streams; ++i) {
      for (auto node_index : nodes_in_stream[i]) {
        auto* node = graph_viewer_.GetNode(node_index);
        for (auto it = node->OutputNodesBegin(); it != node->OutputNodesEnd(); ++it) {
          if (std::find(nodes_in_stream[i].begin(), nodes_in_stream[i].end(), it->Index()) == nodes_in_stream[i].end()) {
            node_to_notification[node_index] = num_notifications++;
            break;
          }
        }
      }
    }
    //3. Check the nodes in each logical stream, build the map;
    for (auto i = 0; i < num_logic_streams; ++i) {
      std::set<const IExecutionProvider*> providers;
      for (auto node_index : nodes_in_stream[i]) {
        node_to_stream_map[node_index] = node_stream_map[node_index];
        auto* node = graph_viewer_.GetNode(node_index);
        onnxruntime::ProviderType exec_provider_name = node->GetExecutionProviderType();
        const IExecutionProvider* ep = execution_providers_.Get(exec_provider_name);
        if (plan.logic_streams_[node_stream_map[node_index]]->ep_) {
          ORT_ENFORCE(plan.logic_streams_[node_stream_map[node_index]]->ep_ == ep);
        } else {
          plan.logic_streams_[node_stream_map[node_index]]->ep_ = ep;
        }
      }
    }
    //4. set notification owners
    plan.notification_owners_.resize(num_notifications);
    for (auto node_index : graph_viewer_.GetNodesInTopologicalOrder()) {
      auto it = node_to_notification.find(node_index);
      if (it != node_to_notification.end()) {
        // notification owned by the node who produced it.
        plan.notification_owners_[it->second] = node_stream_map[node_index];
      }
    }
    //5. add commands to logic queue
    for (auto i = 0; i < num_logic_streams; ++i) {
      for (auto j = 0; j < nodes_in_stream[i].size(); ++j) {
        auto node_index = nodes_in_stream[i][j];
        if (j > 0) {
          // add dependency for current logic stream
          dependence_graph_[node_index].insert(nodes_in_stream[i][j - 1]);
        }
        auto cur_stream_idx = node_to_stream_map[node_index];
        // check if any producer is not in current stream, if yes, create a wait
        auto* node = graph_viewer_.GetNode(node_index);
        for (auto it = node->InputNodesBegin(); it != node->InputNodesEnd(); ++it) {
          if (std::find(nodes_in_stream[i].begin(), nodes_in_stream[i].end(), it->Index()) == nodes_in_stream[i].end()) {
            // find the notificaiton id
            auto notfication_it = node_to_notification.find(it->Index());
            ORT_ENFORCE(notfication_it != node_to_notification.end());
            NotificationIndex notification_index = notfication_it->second;
            // push a barrier
            int barrier_id = plan.num_barriers_++;
            plan.downstream_map_[notification_index].push_back({i, static_cast<int>(plan.logic_streams_[i]->commands_.size())});
            plan.logic_streams_[i]->commands_.push_back([barrier_id](void* ctx, bool& continue_flag) {
              ExecutionPlanContext* execution_context = reinterpret_cast<ExecutionPlanContext*>(ctx);
              continue_flag = execution_context->DecCountDownBarrier(barrier_id);
              return Status::OK();
            });
            // push a wait command if has EP registered it.
            auto wait_handle = stream_handle_registry_.GetWaitHandle(
                plan.logic_streams_[plan.notification_owners_[notfication_it->second]]->ep_->Type(),
                node->GetExecutionProviderType());
            if (wait_handle) {
              const std::string& upstream_node_name = it->Name();
              plan.logic_streams_[i]->commands_.push_back([wait_handle, cur_stream_idx, notification_index, i, node, upstream_node_name](void* ctx, bool& continue_flag) {
                ExecutionPlanContext* execution_context = reinterpret_cast<ExecutionPlanContext*>(ctx);
                wait_handle(*execution_context->GetDeviceStream(cur_stream_idx), *execution_context->notifications_[notification_index]);
                // update streams clock status
                if (execution_context->GetDeviceStream(cur_stream_idx)) {
                  execution_context->GetDeviceStream(cur_stream_idx)->UpdateStreamClock(execution_context->notifications_[notification_index]->stream_clock_);
                }
                LOGS(execution_context->logger_, INFO) << "stream " << i << " wait on " << upstream_node_name << " for " << node->Name();
                continue_flag = true;
                return Status::OK();
              });
            }
          }
        }
        for (auto it = node->OutputNodesBegin(); it != node->OutputNodesEnd(); ++it) {
          // add dependency for model graph
          dependence_graph_[it->Index()].insert(node_index);
        }
        // push launch kernel command
        plan.logic_streams_[i]->commands_.push_back([this, node, node_index, cur_stream_idx, i](void* ctx, bool& continue_flag) {
          ExecutionPlanContext* execution_context = reinterpret_cast<ExecutionPlanContext*>(ctx);
          ORT_RETURN_IF_ERROR(execution_context->ExecuteKernel(node_index, cur_stream_idx));
          return Status::OK();
        });
        // check if any notification generated by this node, if yes, push a activate
        auto notification_it = node_to_notification.find(node_index);
        if (notification_it != node_to_notification.end()) {
          NotificationIndex notification_index = notification_it->second;
          plan.logic_streams_[i]->commands_.push_back([notification_index, i, node](void* ctx, bool& continue_flag) {
            ExecutionPlanContext* execution_context = reinterpret_cast<ExecutionPlanContext*>(ctx);
            if (execution_context->notifications_[notification_index]) {
              execution_context->notifications_[notification_index]->ActivateAndUpdate();
            }
            LOGS(execution_context->logger_, INFO) << "stream " << i << " send notification for " << node->Name();
            continue_flag = true;
            return Status::OK();
          });
          // notify downstreams
          plan.logic_streams_[i]->commands_.push_back([this, notification_index, &plan](void* ctx, bool& continue_flag) {
            ExecutionPlanContext* execution_context = reinterpret_cast<ExecutionPlanContext*>(ctx);
            plan.ScheduleDownstream(*execution_context, notification_index);
            continue_flag = true;
            return Status::OK();
          });
        }
      }
    }
    // 6. now prepare for release plan
    int num_ml_values = ort_value_name_idx_map_.MaxIdx() + 1;
    std::unordered_set<int> node_outputs;
    value_ref_counts_.resize(num_ml_values);

    for (auto node_index : graph_viewer_.GetNodesInTopologicalOrder()) {
      auto* node = graph_viewer_.GetNode(node_index);
      const auto& output_defs = node->OutputDefs();
      for (int output_idx_local = 0; output_idx_local < output_defs.size(); ++output_idx_local) {
        const auto& node_output = output_defs[output_idx_local];
        if (!node_output->Exists()) continue;
        OrtValueIndex output_idx_global;
        ORT_THROW_IF_ERROR(ort_value_name_idx_map_.GetIdx(node_output->Name(), output_idx_global));
        node_outputs.insert(output_idx_global);
        plan.value_to_stream_map_[output_idx_global] = node_to_stream_map[node_index];
        value_node_map_[output_idx_global] = node_index;
      }
    }

    for (auto node_index : graph_viewer_.GetNodesInTopologicalOrder()) {
      auto* node = graph_viewer_.GetNode(node_index);
      const auto& input_node_args = node->InputDefs();
      for (int input_index_local = 0; input_index_local < input_node_args.size(); ++input_index_local) {
        const auto* input_arg = input_node_args[input_index_local];
        //skip optional inputs
        if (input_arg->Exists()) {
          OrtValueIndex input_idx_global;
          ORT_THROW_IF_ERROR(ort_value_name_idx_map_.GetIdx(input_arg->Name(), input_idx_global));
          plan.value_consumer_map_[input_idx_global].insert(node_index);
          if (node_outputs.find(input_idx_global) != node_outputs.end()) {
            value_ref_counts_[input_idx_global]++;
            node_value_map[node_index].push_back(input_idx_global);
          }
        }
      }
    }

    plan.allocation_plan_.resize(num_ml_values);

    // 7. set per-value memory location
    if (!SetValueLocation(plan).IsOK()) {
      ORT_THROW("Failed to set local for each ort value");
    }

    // 8. set per-value alloc plan
    if (!SetAllocPlan(plan).IsOK()) {
      ORT_THROW("Failed to set alloc plan for each ort value");
    }

    // 9. try reuse tensor
    TryReuseTensor(plan);

    //TODO: move this to logger
    /*for (int i = 0; i < streams_log.size(); ++i) {
      if (streams_log[i].empty()) {
        std::cout << "stream " << i << ": <empty>" << std::endl;
      } else {
        std::stringstream ss;
        std::copy(streams_log[i].begin(), streams_log[i].end() - 1, std::ostream_iterator<std::string>(ss, ","));
        ss << streams_log[i].back();
        std::cout << "stream " << i << ": " << ss.str() << std::endl;
      }
    }*/

    return onnxruntime::Status::OK();
  }

  static const KernelCreateInfo& GetKernelCreateInfo(
      const KernelCreateInfoMap& kernel_create_info_map,
      NodeIndex node_index) {
    auto entry = kernel_create_info_map.find(node_index);
    ORT_ENFORCE(entry != kernel_create_info_map.cend(),
                "SessionState should have saved the KernelCreateInfo prior to this running. NodeIndex:", node_index);

    return *entry->second;
  }

  OrtValueIndex Index(const OrtValueName& name) {
    OrtValueIndex result;
    auto status = ort_value_name_idx_map_.GetIdx(name, result);
    ORT_ENFORCE(status.IsOK(), status.ErrorMessage());
    return result;
  }

  int& UseCount(OrtValueIndex n) {
    ORT_ENFORCE(n >= 0 && n <= ort_value_name_idx_map_.MaxIdx());
    return value_ref_counts_[n];
  }

  int& UseCount(const OrtValueName& name) { return UseCount(Index(name)); }

  bool HasExternalOutputs(const Node& node) const {
    const KernelCreateInfo& ci = GetKernelCreateInfo(kernel_create_info_map_, node.Index());
    if (ci.kernel_def == nullptr) {
      return false;
    }
    return ci.kernel_def->HasExternalOutputs();
  }

  Status SetValueLocation(ExecutionPlanImpl& plan) {
    // Note: for every ml-value, its definition must appear before all its uses in a topological sort of a valid model
    using GraphInputsSet = InlinedHashSet<std::string_view>;
    const auto& graph_inputs_nodes = graph_viewer_.GetInputsIncludingInitializers();
    GraphInputsSet graph_inputs;
    graph_inputs.reserve(graph_inputs_nodes.size());
    for (auto& graph_input : graph_inputs_nodes) {
      graph_inputs.insert(graph_input->Name());
    }

    for (auto graph_input : graph_viewer_.GetInputs()) {
      OrtValueIndex index = Index(graph_input->Name());
      UseCount(index)++;  // Models caller's usage post-inference; ensures it will not be reused.
    }

    for (auto node_arg : outer_scope_node_args_) {
      OrtValueIndex index = Index(node_arg->Name());
      UseCount(index)++;  // ensure will not be re-used as this graph does not own the buffer
    }

    // All initializers should be treated as input
    for (const auto& pair : graph_viewer_.GetAllInitializedTensors()) {
      const auto& initializer_name = pair.first;
      OrtValueIndex index = Index(initializer_name);
      UseCount(initializer_name)++;
    }

    InlinedHashSet<OrtValueIndex> set_node_arg_has_explicit_consumer;

    InlinedHashMap<OrtValueIndex, const IExecutionProvider*> map_implicitly_consumed_node_arg_to_ep;
    InlinedHashSet<OrtValueIndex> set_implicitly_consumed_node_arg_has_heterogenous_ep_consumers;

    for (auto node_index : graph_viewer_.GetNodesInTopologicalOrder()) {
      auto pnode = graph_viewer_.GetNode(node_index);
      if (pnode == nullptr) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Can not find the node ", node_index);
      }

      // Identify where each output of this node should be allocated.
      // This is determined by the OpKernel bound to the node.
      const KernelCreateInfo& kernel_create_info = GetKernelCreateInfo(kernel_create_info_map_, pnode->Index());
      const auto* p_kernel_def = kernel_create_info.kernel_def.get();
      ORT_ENFORCE(p_kernel_def, "Should not have entry in kernel create info with nullptr for kernel_def");
      auto exec_provider = execution_providers_.Get(*pnode);
      if (exec_provider == nullptr) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Can not find the execution provider ",
                               pnode->GetExecutionProviderType());
      }

      bool is_implicit_input = false;
      // increment UseCount and add location information if applicable for the provided input def
      auto process_input = [&plan, &graph_inputs, &exec_provider, &p_kernel_def, &is_implicit_input,
                            &set_node_arg_has_explicit_consumer,
                            &map_implicitly_consumed_node_arg_to_ep,
                            &set_implicitly_consumed_node_arg_has_heterogenous_ep_consumers,
                            this](const NodeArg& input, size_t arg_idx) {
        const auto& name = input.Name();
        UseCount(name)++;
        bool is_graph_input = (graph_inputs.find(name) != graph_inputs.cend());
        bool is_outer_scope_arg = std::find_if(outer_scope_node_args_.cbegin(), outer_scope_node_args_.cend(),
                                               [&name](const NodeArg* value) {
                                                 return value && value->Name() == name;
                                               }) != outer_scope_node_args_.cend();
        bool is_subgraph = (parent_node_ != nullptr);

        // If it's a graph input or outer scope node arg, set its plan.
        // NOTE: Copy nodes should have already been added if a graph input is fed as input
        // to nodes assigned to different providers.

        if (is_graph_input || is_outer_scope_arg) {
          OrtValueIndex index = Index(name);

          if (!is_implicit_input) {
            OrtMemType mem_type = p_kernel_def->InputMemoryType(arg_idx);
            //plan_.SetLocation(static_cast<size_t>(index), exec_provider->GetAllocator(0, mem_type)->Info());
            //plan_.SetLocation(static_cast<size_t>(index), exec_provider->GetAllocator(0, mem_type)->Info());
            plan.allocation_plan_[index].location = exec_provider->GetAllocator(0, mem_type)->Info();
            set_node_arg_has_explicit_consumer.insert(index);
          } else {  // implicit input
            // Only process an implicit input if there are explicit consumers at this graph level
            // If there is an explicit consumer, the location MUST be where it is consumed
            // and not where it is located in the outer scope.
            // It is okay if we process a node consuming this arg as an implicit input
            // ahead of a node that is an explicit consumer, because we will just reset
            // this location in the 'if' branch above.

            // CASE 1: We see an implicit input without explicit consumers in a subgraph (pass-through subgraph inputs),
            // then set its location to be its corresponding location in the outer scope.
            // This is so that the subgraph copying mechanism doesn't trigger an unnecessary copy and any copying
            // decisions are deferred till there is an explicit consumer of the subgraph input in nested subgraphs.
            if (is_subgraph && set_node_arg_has_explicit_consumer.count(index) == 0) {
              auto iter = outer_scope_node_arg_to_location_map_.find(name);
              bool found_in_outer_scope_location_map = (iter != outer_scope_node_arg_to_location_map_.end());

              if (!is_graph_input) {
                // Failing this enforce for an implicit subgraph input points to an internal error somewhere.
                // For certain older opsets (Scan-8), we may not have added explicit subgraph inputs
                // to the outer scope location map. See explanation in IsNodeWhereNodeInputsAreSameAsExplicitSubgraphInputs()
                // called in FinalizeSessionStateImpl() in SessionState.
                ORT_ENFORCE(found_in_outer_scope_location_map,
                            "There is no location for this node arg in the outer scope location map");
              }

              if (found_in_outer_scope_location_map) {
                plan.allocation_plan_[index].location = iter->second;
              }
            } else if (set_node_arg_has_explicit_consumer.count(index) == 0) {
              // CASE 2: We see an implicit input without explicit consumers in the main graph,
              // then set its location to be the device corresponding to the EP that the subgraph
              // holding node has been partitioned to.

              // The "ideal" solution is to set the location of its first "explicit" usage which may occur
              // in any nested subgraph of the node, but that is potentially too costly to
              // get at this stage (TODO: Investigate feasibility of this, see TODO in FinalizeSessionStateImpl() around this)

              // Instead, we take a "less than ideal" route which is to set the location to be the device
              // corresponding to the EP that the node is partitioned to. The hypothesis is that it is "most likely"
              // that the implicit input will eventually be consumed on that device in a nested subgraph.

              // The previous behavior was to default to CPU which will cause unnecessary copies when
              // (1) The user invokes Run() with an OrtValue backed by non-CPU memory (eg CUDA) and
              // the node in the subgraph that consumes the subgraph's implicit input is on a non-CPU device
              // in the subgraph
              // (2) The user tries to IOBind implicitly consumed graph inputs (GH Issue 11254) and
              // the node in the subgraph that consumes the subgraph's implicit input is on
              // a non-CPU device in the subgraph

              // Even if the user provides an input on CPU and the node in the subgraph that consumes the subgraph's
              // implicit input is on a non-CPU device, instead of the subgraph copying mechanism taking it to the device,
              // all we will do is "front-load" this copy in utils::CopyInputsAcrossDevices() with this approach.

              // NOTE 1: The only case this will be sub-optimal is when a node containing a subgraph is partitioned to a
              // non-CPU EP and the user provides an input (or tries to IOBind the input) AND it will eventually be
              // explicitly consumed on CPU - this scenario should be very rare and we forgo performance in this case
              // (the subgraph copying mechanism will make the copy to CPU eventually) in favor of optimizing for the
              // common case (which is that we expect the implicit input to be consumed on the non-CPU device corresponding
              // to the non-CPU EP).

              // NOTE 2: If the implicit input is consumed by multiple nodes (as implicit inputs in all of them) and
              // all of them are partitioned to the same EP, then we go ahead with the above stated logic.
              // If there are multiple EPs involved, we default the location to just CPU as there is ambiguity involved
              // as to which non-CPU device is "most optimal" for the implicit input.

              if (set_implicitly_consumed_node_arg_has_heterogenous_ep_consumers.count(index) == 0) {
                auto already_seen_ep_for_node_arg = map_implicitly_consumed_node_arg_to_ep.find(index);

                if (already_seen_ep_for_node_arg == map_implicitly_consumed_node_arg_to_ep.end()) {
                  // First time we are encountering this implicitly consumed input at this graph level (or)
                  // plan_.SetLocation(static_cast<size_t>(index), exec_provider->GetAllocator(0, OrtMemType::OrtMemTypeDefault)->Info());
                  plan.allocation_plan_[index].location = exec_provider->GetAllocator(0, OrtMemType::OrtMemTypeDefault)->Info();
                  map_implicitly_consumed_node_arg_to_ep.insert({index, exec_provider});
                } else if (already_seen_ep_for_node_arg->second == exec_provider) {
                  // The EP that we previously seen for this implicit input is the same one as the current EP
                  // we have seen
                  // plan_.SetLocation(static_cast<size_t>(index), exec_provider->GetAllocator(0, OrtMemType::OrtMemTypeDefault)->Info());
                  plan.allocation_plan_[index].location = exec_provider->GetAllocator(0, OrtMemType::OrtMemTypeDefault)->Info();
                } else {
                  // Default the location to CPU
                  // plan_.SetLocation(static_cast<size_t>(index), execution_providers_.Get(CPU)->GetAllocator(0, OrtMemType::OrtMemTypeDefault)->Info());
                  plan.allocation_plan_[index].location = execution_providers_.Get(CPU)->GetAllocator(0, OrtMemType::OrtMemTypeDefault)->Info();
                  set_implicitly_consumed_node_arg_has_heterogenous_ep_consumers.insert(index);
                }
              }
            }
          }
        }

        return Status::OK();
      };

      ORT_RETURN_IF_ERROR(Node::ForEachWithIndex(pnode->InputDefs(), process_input));

      is_implicit_input = true;
      ORT_RETURN_IF_ERROR(Node::ForEachWithIndex(pnode->ImplicitInputDefs(), process_input));

      auto outputs = pnode->OutputDefs();
      auto num_outputs = outputs.size();
      bool has_external_outputs = HasExternalOutputs(*pnode);
      for (size_t i = 0; i < num_outputs; ++i) {
        auto* node_output = outputs[i];
        if (!node_output->Exists()) continue;
        OrtValueIndex index = Index(node_output->Name());
        // Ensures external outputs will not be reused.
        UseCount(index) += (has_external_outputs ? 2 : 1);
        auto allocator = exec_provider->GetAllocator(0, p_kernel_def->OutputMemoryType(i));
        ORT_ENFORCE(allocator);
        // plan_.SetLocation(static_cast<size_t>(index), allocator->Info());
        plan.allocation_plan_[index].location = allocator->Info();
      }
    }

    for (auto graph_output : graph_viewer_.GetOutputs()) {
      UseCount(graph_output->Name())++;  // Models caller's usage post-inference; ensures it will not be reused.
    }

    return Status::OK();
  }

  OrtMemoryInfo GetLocationForNodeInput(size_t input_index, const Node& node,
                                        const KernelCreateInfoMap& kernel_create_info_map) {
    auto* p_provider = execution_providers_.Get(node);
    ORT_ENFORCE(p_provider);
    const KernelCreateInfo& kernel_create_info = GetKernelCreateInfo(kernel_create_info_map, node.Index());
    if (utils::IsInputOnCpu(node, &kernel_create_info, input_index))
      // weights are not output from any node, so it's OK to put its location on CPU provider
      return execution_providers_.GetDefaultCpuMemoryInfo();
    return p_provider->GetAllocator(0, OrtMemTypeDefault)->Info();
  }

  void GeneratePlanForWeightsHelper(const GraphViewer& graph_viewer,
                                    const InitializedTensorSet& weights,
                                    const KernelCreateInfoMap& kernel_create_info_map,
                                    const std::string& subgraph_kernel_create_info_map_key_base,
                                    size_t graph_depth,
                                    /*out*/ std::vector<std::vector<OrtMemoryInfo>>& locations) {
    // Iterate over nodes in current level firstly to record location of usages
    // in current graph
    for (const auto& node : graph_viewer.Nodes()) {
      const auto& input_node_args = node.InputDefs();
      size_t num_node_inputs = input_node_args.size();

      for (size_t node_input_index = 0; node_input_index < num_node_inputs; ++node_input_index) {
        auto input_node_arg = input_node_args[node_input_index];
        // Skip processing missing optional inputs
        if (!input_node_arg->Exists()) {
          continue;
        }

        auto& def_name = input_node_arg->Name();
        // This node input doesn't correspond to any of the weights
        if (!weights.count(def_name)) {
          continue;
        }

        // While processing subgraphs, if we don't see an entry in the implicit
        // inputs of the node containing the subgraph, it is a shadow value.
        auto is_shadow_value_in_subgraph = [](const Node& subgraph_parent_node,
                                              const std::string& def_name) -> bool {
          bool is_shadow_value_in_subgraph = true;
          for (const auto& implicit_input : subgraph_parent_node.ImplicitInputDefs()) {
            if (implicit_input->Name() == def_name) {
              is_shadow_value_in_subgraph = false;
              break;
            }
          }

          return is_shadow_value_in_subgraph;
        };

        // Skip processing shadow values in subgraphs
        if (graph_depth > 0) {
          // We are processing a subgraph if we enter this
          const auto* parent_node = graph_viewer.ParentNode();

          // Skip processing if it is a shadow value
          if (is_shadow_value_in_subgraph(*parent_node, def_name)) {
            continue;
          }
        }

        auto wt_index = Index(def_name);
        // TODO: Identify error cases where-in an initializer is used on different
        // devices within the same graph level.
        // If we ever encounter that, it means that there is a severe bug in Memcpy
        // transformer and the model will crash while running. The Memcpy transformer
        // is supposed to duplicate initializers being used on different devices within
        // the same graph level and hence we should never see an initializer being used
        // on different devices here.
        // The same initializer being used on different devices across graph levels
        // (subgraphs) is okay and utils::CopyInputsAcrossDevices() will take it to
        // the right device before subgraph execution.
        locations[wt_index].emplace_back(
            GetLocationForNodeInput(node_input_index, node, kernel_create_info_map));
      }
    }

    // Iterate over nodes in current graph with subgraphs and recurse.
    for (const auto& node : graph_viewer.Nodes()) {
      // If the node has subgraphs (i.e.) control flow nodes,
      // walk the nodes in those subgraphs as well to best determine
      // the location for the OrtValue corresponding to the weights
      // (i.e.) do a recursion
      if (node.ContainsSubgraph()) {
        // A node may contain multiple subgraphs - so iterate through all of them
        for (auto& name_to_subgraph : node.GetAttributeNameToSubgraphMap()) {
          GraphViewer subgraph_viewer(*name_to_subgraph.second);

          const auto& local_subgraph_kernel_create_info_map_key =
              NestedSubgraphInfoDetails::ComposeNestedSubgraphInfoKeyHelper(subgraph_kernel_create_info_map_key_base,
                                                                            graph_depth, node.Index(), name_to_subgraph.first);

          auto specific_subgraph_kernel_create_info_map = subgraphs_kernel_create_info_maps_.find(local_subgraph_kernel_create_info_map_key);
          ORT_ENFORCE(specific_subgraph_kernel_create_info_map != subgraphs_kernel_create_info_maps_.end());

          GeneratePlanForWeightsHelper(subgraph_viewer,
                                       weights,
                                       specific_subgraph_kernel_create_info_map->second,
                                       local_subgraph_kernel_create_info_map_key,
                                       graph_depth + 1,
                                       locations);
        }
      }
    }
  }

  Status GeneratePlanForWeights(ExecutionPlanImpl& plan) {
    std::vector<std::vector<OrtMemoryInfo>> locations(plan.allocation_plan_.size());
    GeneratePlanForWeightsHelper(graph_viewer_,
                                 graph_viewer_.GetAllInitializedTensors(),
                                 kernel_create_info_map_,
                                 "", 0, locations);
    for (size_t i = 0; i != locations.size(); ++i) {
      const std::vector<OrtMemoryInfo>& loc = locations[i];
      if (loc.empty()) continue;
      plan.allocation_plan_[i].alloc_kind = AllocKind::kAllocateStatically;
      plan.allocation_plan_[i].location = loc[0];
    }
    return Status::OK();
  }

  static bool IsNonTensor(const onnxruntime::NodeArg& nodearg) {
    // TODO: unclear why we should go through a string-representation of type
    auto ptype = nodearg.Type();
    auto& type_proto = ONNX_NAMESPACE::Utils::DataTypeUtils::ToTypeProto(ptype);
    return !utils::HasTensorType(type_proto);
  }

  onnxruntime::Status SetAllocPlan(ExecutionPlanImpl& plan) {
    auto setup_preexisting = [this, &plan](const NodeArg* node_arg) {
      auto input_index = Index(node_arg->Name());
      AllocPlanPerValue& thisplan = plan.allocation_plan_[input_index];
      thisplan.alloc_kind = AllocKind::kPreExisting;
      thisplan.value_type = utils::GetMLDataType(*node_arg);
    };

    for (auto graph_input : graph_viewer_.GetInputs()) {
      setup_preexisting(graph_input);
    }

    // outer scope node args are treated the same as graph inputs
    for (auto outer_scope_node_arg : outer_scope_node_args_) {
      setup_preexisting(outer_scope_node_arg);
    }

    // set AllocationInfo for each weight
    ORT_RETURN_IF_ERROR(GeneratePlanForWeights(plan));
    const auto& graph_outputs = graph_viewer_.GetOutputs();
    for (auto node_index : graph_viewer_.GetNodesInTopologicalOrder()) {
      const auto* pnode = graph_viewer_.GetNode(node_index);
      // node outputs.
      const auto& output_defs = pnode->OutputDefs();
      // External outputs flag.
      bool has_external_outputs = HasExternalOutputs(*pnode);
      for (size_t output_arg_def_index = 0, end = output_defs.size(); output_arg_def_index < end; ++output_arg_def_index) {
        const auto& node_output = output_defs[output_arg_def_index];
        if (!node_output->Exists()) continue;
        // OrtValue index of the considered output NodeArg.
        const auto current = Index(node_output->Name());
        plan.allocation_plan_[current].value_type = utils::GetMLDataType(*node_output);
        if (has_external_outputs) {
          ORT_ENFORCE(!IsNonTensor(*node_output), "Only tensors are supported for external outputs for now.");
          plan.allocation_plan_[current].alloc_kind = AllocKind::kAllocatedExternally;
        } else if (std::find(graph_outputs.begin(), graph_outputs.end(), node_output) != graph_outputs.end()) {
          plan.allocation_plan_[current].alloc_kind = AllocKind::kAllocateOutput;
          // hacky perf optimization, bla bla bla ...
          // is there a UT for it?
          if (parent_node_ && pnode->OpType() == "Identity" && parent_node_->OpType() == "Loop") {
            const NodeArg* input = pnode->InputDefs()[0];
            bool input_is_loop_iteration_number = input == graph_viewer_.GetInputs()[0];
            if (!input_is_loop_iteration_number) {
              const auto& input_name = input->Name();
              const auto input_index = Index(input_name);
              const auto& alloc_plan = plan.allocation_plan_[input_index];
              if (alloc_plan.alloc_kind == AllocKind::kPreExisting) {
                plan.allocation_plan_[current].alloc_kind = AllocKind::kShare;
                plan.allocation_plan_[current].reused_buffer = input_index;
              }
            }
          }
        } else {
          plan.allocation_plan_[current].alloc_kind = AllocKind::kAllocate;
        }
      }
    }
    return Status::OK();
  }

  void TryReuseTensor(ExecutionPlanImpl& plan) {
    InlinedHashMap<NodeIndex, int> dependents;
    for (const auto& it : dependence_graph_) {
      for (NodeIndex node_index : it.second) {
        dependents[node_index]++;
      }
    }
    std::deque<NodeIndex> que;
    for (const auto& it : dependence_graph_) {
      if (dependents[it.first] == 0) {
        que.push_back(it.first);
      }
    }

    // fetch_all_dependents will collect all dependent nodes for "node_index"
    std::function<std::set<NodeIndex>(NodeIndex)> fetch_all_dependents = [&](NodeIndex node_index) {
      std::set<NodeIndex> dependents;

      std::function<void(NodeIndex)> dfs = [&](NodeIndex curr) {
        if (dependents.find(curr) == dependents.end()) {
          dependents.insert(curr);
          for (NodeIndex dep : dependence_graph_[curr]) {
            dfs(dep);
          }
        }
      };

      dfs(node_index);
      return dependents;
    };

    // waiting_list keeps all values who want to reuse some upstream values' memory
    std::map<OrtMemoryInfo, std::map<size_t, typename std::map<const onnxruntime::NodeArg* const, std::set<NodeIndex>*>>> waiting_list;

    // for each node, dependents_map keeps all its dependent upstream nodes that are sure to be completed ahead
    std::map<NodeIndex, std::set<NodeIndex>> dependents_map;

    std::map<OrtValueIndex, std::set<OrtValueIndex>> input_output_map;

    std::set<OrtValueIndex> reused;

    //const auto& graph_viewer = impl_->session_state_.GetGraphViewer();
    //const auto& value_map = impl_->session_state_.GetOrtValueNameIdxMap();
    //const auto& kernel_create_info_map = impl_->session_state_.GetKernelCreateInfoMap();
    //const auto& allcation_plan = this->allocation_plan;

    std::function<void(NodeIndex)> TryReuseInput = [&](NodeIndex node_index) {
      auto* node = graph_viewer_.GetNode(node_index);

      for (int output_arg_num = 0; output_arg_num < node->OutputDefs().size(); output_arg_num++) {
        auto p_output_arg = node->OutputDefs()[output_arg_num];
        OrtValueIndex output_idx_global{};

        if (!ort_value_name_idx_map_.GetIdx(p_output_arg->Name(), output_idx_global).IsOK() ||
            plan.allocation_plan_[output_idx_global].alloc_kind != AllocKind::kAllocate) {
          continue;
        }

        auto kci_it = kernel_create_info_map_.find(node_index);
        if (kci_it == kernel_create_info_map_.end()) {
          continue;
        }

        const KernelCreateInfo& ci = *kci_it->second;
        if (ci.kernel_def == nullptr) {
          continue;
        }

        bool found_reusable = false;
        const auto& alias_map = ci.kernel_def->Alias();
        auto input_args = node->InputDefs();
        for (auto* input_arg : input_args) {
          OrtValueIndex input_idx_global{};
          if (ort_value_name_idx_map_.GetIdx(input_arg->Name(), input_idx_global).IsOK()) {
            input_output_map[input_idx_global].insert(output_idx_global);
          }
        }

        for (auto& pair : alias_map) {
          if (pair.second == output_arg_num) {
            // we _must_ reuse this input to satisfy aliasing requirement: (e.g., for reshape)
            if ((0 <= pair.first) && (static_cast<size_t>(pair.first) < input_args.size())) {
              auto p_input_arg = input_args[pair.first];
              if (p_input_arg->Exists()) {
                OrtValueIndex reusable_input{};
                if (ort_value_name_idx_map_.GetIdx(p_input_arg->Name(), reusable_input).IsOK() &&
                    plan.allocation_plan_[reusable_input].alloc_kind == AllocKind::kAllocate) {
                  // LOGS(const_cast<SessionState&>(impl_->session_state_).Logger(), INFO) << p_input_arg->Name() << " reused by " << p_output_arg->Name() << " as input" << std::endl;
                  std::cout << p_input_arg->Name() << " reused by " << p_output_arg->Name() << " as input" << std::endl;
                  plan.allocation_plan_[output_idx_global].alloc_kind = AllocKind::kReuse;
                  plan.allocation_plan_[output_idx_global].reused_buffer = reusable_input;
                  plan.value_consumer_map_[reusable_input].insert(plan.value_consumer_map_[output_idx_global].begin(),
                                                                  plan.value_consumer_map_[output_idx_global].end());
                  reused.insert(reusable_input);
                  found_reusable = true;
                  break;
                }
              }
            }
          }
        }

        if (found_reusable) {
          continue;
        }

        const auto& variadic_alias_offsets = ci.kernel_def->VariadicAlias();
        if (variadic_alias_offsets.has_value()) {
          int input_offset = variadic_alias_offsets->first;
          int output_offset = variadic_alias_offsets->second;
          int alias_input_index = output_arg_num - output_offset + input_offset;

          if (alias_input_index >= 0 && static_cast<size_t>(alias_input_index) < input_args.size()) {
            auto p_input_arg = input_args[alias_input_index];

            if (p_input_arg->Exists()) {
              OrtValueIndex reusable_input{};
              if (ort_value_name_idx_map_.GetIdx(p_input_arg->Name(), reusable_input).IsOK() &&
                  plan.allocation_plan_[reusable_input].alloc_kind == AllocKind::kAllocate) {
                // LOGS(const_cast<SessionState&>(impl_->session_state_).Logger(), INFO) << p_input_arg->Name() << " reused by " << p_output_arg->Name() << " as input" << std::endl;
                std::cout << p_input_arg->Name() << " reused by " << p_output_arg->Name() << " as input" << std::endl;
                plan.allocation_plan_[output_idx_global].alloc_kind = AllocKind::kReuse;
                plan.allocation_plan_[output_idx_global].reused_buffer = reusable_input;
                plan.value_consumer_map_[reusable_input].insert(plan.value_consumer_map_[output_idx_global].begin(),
                                                                plan.value_consumer_map_[output_idx_global].end());
                reused.insert(reusable_input);
                continue;
              }  //if
            }    //if
          }
        }

        const auto& inplace_map = ci.kernel_def->MayInplace();
        for (auto& pair : inplace_map) {
          if (pair.second == output_arg_num) {
            if ((0 <= pair.first) && (static_cast<size_t>(pair.first) < input_args.size())) {
              auto p_input_arg = input_args[pair.first];
              if (p_input_arg->Exists()) {
                OrtValueIndex input_arg_index{};
                if (ort_value_name_idx_map_.GetIdx(p_input_arg->Name(), input_arg_index).IsOK() &&
                    plan.allocation_plan_[input_arg_index].alloc_kind == AllocKind::kAllocate) {
                  if (plan.value_consumer_map_[input_arg_index].size() == 1 && SameSize(context_, *p_input_arg, *p_output_arg)) {
                    // LOGS(const_cast<SessionState&>(impl_->session_state_).Logger(), INFO) << p_input_arg->Name() << " reused by " << p_output_arg->Name() << " as an input" << std::endl;
                    // std::cout << p_input_arg->Name() << " reused by " << p_output_arg->Name() << " as an input" << std::endl;
                    plan.allocation_plan_[output_idx_global].alloc_kind = AllocKind::kReuse;
                    plan.allocation_plan_[output_idx_global].reused_buffer = input_arg_index;
                    plan.value_consumer_map_[input_arg_index].insert(plan.value_consumer_map_[output_idx_global].begin(),
                                                                     plan.value_consumer_map_[output_idx_global].end());
                    reused.insert(input_arg_index);
                  }
                }
              }
            }
          }
        }
      }
    };  //TryReuseInput

    // go over the outputs of "node_index" and try to reuse its memory
    std::function<void(NodeIndex)> TryReuseOutput = [&](NodeIndex node_index) {
      dependents_map[node_index] = fetch_all_dependents(node_index);
      auto* node = graph_viewer_.GetNode(node_index);
      const auto& output_defs = node->OutputDefs();

      for (int output_idx_local = 0; output_idx_local < output_defs.size(); ++output_idx_local) {
        const auto& node_output = output_defs[output_idx_local];
        if (!node_output->Exists()) continue;
        OrtValueIndex output_idx_global{};

        if (ort_value_name_idx_map_.GetIdx(node_output->Name(), output_idx_global).IsOK()) {
          if (reused.find(output_idx_global) != reused.end() ||
              plan.allocation_plan_[output_idx_global].alloc_kind != AllocKind::kAllocate) {
            continue;  // skip when it is already reused
          }

          const auto* shape = context_.GetShape(*node_output);
          if (!shape) continue;
          size_t size_in_bytes = shape->ByteSizeLong();

          const auto& location = plan.allocation_plan_[output_idx_global].location;
          auto local_iter = waiting_list.find(location);
          if (local_iter == waiting_list.end()) {
            waiting_list[location][size_in_bytes][node_output] = &dependents_map[node_index];
            continue;
          }

          auto size_iter = local_iter->second.find(size_in_bytes);
          if (size_iter == local_iter->second.end()) {
            waiting_list[location][size_in_bytes][node_output] = &dependents_map[node_index];
            continue;
          }

          bool get_reused = false;
          for (auto node_iter = size_iter->second.begin(); node_iter != size_iter->second.end();) {
            const onnxruntime::NodeArg* const downstream_arg = node_iter->first;
            OrtValueIndex downstream_value{};

            if (!ort_value_name_idx_map_.GetIdx(downstream_arg->Name(), downstream_value).IsOK()) {
              node_iter = next(node_iter);
              continue;
            }

            // skip if it is a pair of input and output
            if (input_output_map[output_idx_global].find(downstream_value) != input_output_map[output_idx_global].end()) {
              node_iter = next(node_iter);
              continue;
            }

            const auto* downstream_shape = context_.GetShape(*downstream_arg);
            //if (!(*downstream_shape == *shape)) {
            //  node_iter = next(node_iter);
            //  continue;
            //}
            if (!SameSize(*downstream_shape, *downstream_arg, *shape, *node_output)) {
              node_iter = next(node_iter);
              continue;
            }

            auto* deps = node_iter->second;

            if (deps->find(node_index) == deps->end()) {
              node_iter = next(node_iter);
              continue;
            }

            bool all_covered = true;
            for (auto consumer : plan.value_consumer_map_[output_idx_global]) {
              if (deps->find(consumer) == deps->end()) {
                all_covered = false;
                break;
              }
            }
            if (all_covered) {
              // LOGS(const_cast<SessionState&>(impl_->session_state_).Logger(), INFO) << node_output->Name() << " reused by " << downstream_arg->Name() << " as remote tensor" << std::endl;
              // std::cout << node_output->Name() << " reused by " << downstream_arg->Name() << " as remote tensor" << std::endl;
              plan.allocation_plan_[downstream_value].alloc_kind = AllocKind::kReuse;
              plan.allocation_plan_[downstream_value].reused_buffer = output_idx_global;
              get_reused = true;
              // add new consumer for the value to be reused
              plan.value_consumer_map_[output_idx_global].insert(value_node_map_[downstream_value]);
              plan.value_consumer_map_[output_idx_global].insert(plan.value_consumer_map_[downstream_value].begin(),
                                                                 plan.value_consumer_map_[downstream_value].end());
              node_iter = size_iter->second.erase(node_iter);
              if (size_iter->second.empty()) {
                local_iter->second.erase(size_iter);
              }
              break;  // only resued once
            } else {
              // dependents not fully covered, cannot reuse, try next one in waiting_list
              node_iter = next(node_iter);
            }
          }  // for
          if (get_reused) {
            reused.insert(output_idx_global);
          } else {
            // if not getting reused, add to waiting
            waiting_list[location][size_in_bytes][node_output] = &dependents_map[node_index];
          }
        }
      }
    };  // TryReuseOutput

    // topological traverse of the dependency graph
    std::unordered_set<NodeIndex> visited;
    while (!que.empty()) {
      NodeIndex node_index = que.front();
      visited.insert(node_index);
      TryReuseInput(node_index);   // try reuse node's inputs as its outputs
      TryReuseOutput(node_index);  // try reuse node's outputs for downstream nodes
      que.pop_front();
      for (NodeIndex next_node_index : dependence_graph_[node_index]) {
        if (--dependents[next_node_index] == 0) {
          que.push_back(next_node_index);
        }
      }
    }
  }
};  // struct ExecutionPlannerImpl

ExecutionPlanner::ExecutionPlanner(const Node* parent_node,
                                   const onnxruntime::GraphViewer& graph_viewer,
                                   const std::vector<const NodeArg*>& outer_scope_node_args,
                                   const ExecutionProviders& providers,
                                   const KernelCreateInfoMap& kernel_create_info_map,
                                   const SubgraphsKernelCreateInfoMaps& subgraphs_kernel_create_info_maps,
                                   const std::unordered_map<OrtValueName, OrtMemoryInfo>& outer_scope_node_arg_to_location_map,
                                   const OrtValueNameIdxMap& ort_value_name_idx_map,
                                   IStreamCommandHandleRegistry& stream_handle_registry,
                                   const ProviderStreamMap& provider_stream_map,
                                   const OpStreamMap& op_stream_map,
                                   const ISequentialPlannerContext& context) {
  // init planner
  planner_impl_ = std::make_unique<ExecutionPlannerImpl>(parent_node,
                                                         graph_viewer,
                                                         outer_scope_node_args,
                                                         providers,
                                                         kernel_create_info_map,
                                                         subgraphs_kernel_create_info_maps,
                                                         outer_scope_node_arg_to_location_map,
                                                         ort_value_name_idx_map,
                                                         stream_handle_registry,
                                                         provider_stream_map,
                                                         op_stream_map,
                                                         context);
}

ExecutionPlanner::~ExecutionPlanner() {}

//std::unique_ptr<ExecutionPlan> ExecutionPlanner::CreatePlan() {
//  return planner_impl_->CreatePlan();
//}

onnxruntime::Status ExecutionPlanner::CreatePlan(ExecutionPlan& plan) {
  return planner_impl_->CreatePlan(plan);
}

void ExecutionPlanImpl::ScheduleDownstream(ExecutionPlanContext& ctx, onnxruntime::NotificationIndex notification_index) {
  auto* ctx_ptr = &ctx;
  for (auto downstream : downstream_map_[notification_index]) {
    concurrency::ThreadPool::Schedule(ctx.session_state_.GetInterOpThreadPool(), [this, ctx_ptr, downstream]() {
      logic_streams_[downstream.first]->RunSince(*ctx_ptr, downstream.second);
    });
  }
}

void LogicStream::RunSince(ExecutionPlanContext& ctx, size_t since) {
  if (!ctx.task_status_.IsOK()) {
    // already in bad status, terminate it
    ctx.CompleteTask();
    return;
  }
  while (since < commands_.size()) {
    if (ctx.terminate_flag_) {
      ctx.SetStatus(ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Exiting due to terminate flag being set to true."));
      return;
    }
    bool continue_flag = true;
    Status status;
    ORT_TRY {
      status = commands_[since](&ctx, continue_flag);
    }
    ORT_CATCH(const std::exception& ex) {
      ORT_HANDLE_EXCEPTION([&]() {
        status = ORT_MAKE_STATUS(ONNXRUNTIME, RUNTIME_EXCEPTION, ex.what());
      });
    }
    if (!status.IsOK()) {
      //terminate it
      ctx.SetStatus(status);
      ctx.CompleteTask();
      return;
    }
    if (!continue_flag) {
      //break but not terminate
      return;
    }
    since++;
  }
  ctx.CompleteTask();
  return;
}

onnxruntime::Status ExecutionPlanExecutor::Execute(const SessionState&,
                                                   const std::vector<int>&,
                                                   const std::vector<OrtValue>&,
                                                   const std::vector<int>&,
                                                   std::vector<OrtValue>&,
                                                   const std::unordered_map<size_t, IExecutor::CustomAllocator>&,
                                                   const logging::Logger&,
                                                   const DeviceStreamColloection&,
                                                   const bool&,
                                                   const bool) {
  return Status::OK();
}

//onnxruntime::Status ExecutionPlanExecutor::Execute(const SessionState& session_state,
//                                                   const std::vector<int>& feed_mlvalue_idxs,
//                                                   const std::vector<OrtValue>& feeds,
//                                                   const std::vector<int>& fetch_mlvalue_idxs,
//                                                   std::vector<OrtValue>& fetches,
//                                                   const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
//                                                   const logging::Logger& logger,
//                                                   const DeviceStreamColloection& device_streams,
//                                                   const bool& terminate_flag,
//                                                   const bool only_execute_path_to_fetches) {
//  if (only_execute_path_to_fetches) {
//    ORT_THROW("NOT IMPLEMENTED YET.");
//  }
//
//  onnxruntime::TimePoint session_start{};
//  auto is_profiler_enabled = session_state.Profiler().IsEnabled();
//  if (is_profiler_enabled) {
//    session_start = session_state.Profiler().Start();
//  }
//
//  ExecutionPlan& plan = *session_state.GetTheExecutionPlan();
//  ExecutionFrame frame(feed_mlvalue_idxs, feeds, fetch_mlvalue_idxs, fetches, fetch_allocators, session_state, &device_streams.GetStreams());
//  ExecutionPlanContext context(session_state, plan, frame, session_state.Logger(), device_streams, terminate_flag);
//  auto* tp = session_state.GetInterOpThreadPool();
//
//  LOGS(logger, INFO) << "Number of streams: " << plan.NumStreams();
//  LOGS(logger, INFO) << "Begin execution";
//
//  for (int i = 0; i < plan.impl_->logic_streams_.size(); ++i) {
//    if (!plan.impl_->logic_streams_[i]->commands_.empty()) {
//      concurrency::ThreadPool::Schedule(tp, [i, this, &context, &plan]() {
//        plan.impl_->logic_streams_[i]->RunSince(context, 0);
//      });
//    }
//  }
//
//  LOGS(logger, INFO) << "Done execution";
//
//  context.WaitAll();
//  ORT_RETURN_IF_ERROR(context.task_status_);
//  ORT_RETURN_IF_ERROR(frame.GetOutputs(fetches));
//
//  if (is_profiler_enabled) {
//    session_state.Profiler().EndTimeAndRecordEvent(profiling::SESSION_EVENT, "SequentialExecutor::Execute", session_start);
//  }
//  return Status::OK();
//}

}  // namespace onnxruntime