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

struct ExecutionContext;
using CommandFn = std::function<Status(ExecutionContext&, bool&)>;

// a logic stream to execute command.
// each command in the logic stream will be executed in FIFO
// a logic stream will be binded to multiple device stream, as the command in the same logic stream may be executed on different EPs.
// i.e., if we set concurrency level to 1, the single logic stream will be equal to our sequential execution plan, which has both cpu and gpu kernels
struct LogicStream {
  std::vector<CommandFn> commands_;
  const IExecutionProvider* ep_ = nullptr;
  void RunSince(ExecutionContext& ctx, size_t since);
  ~LogicStream() {}
};

struct ReleasePlan {
  std::unique_ptr<std::atomic_int[]> value_ref_counts_;
  std::unordered_map<OrtValueIndex, OrtValueIndex> reused_map_;
  std::unordered_map<onnxruntime::NodeIndex, std::vector<OrtValueIndex>> node_value_map_;
};

class DeviceStreamColloectionImpl {
 public:
  DeviceStreamColloectionImpl(size_t num_streams) : num_streams_(num_streams) {
    device_streams_.resize(num_streams, nullptr);
  }

  virtual ~DeviceStreamColloectionImpl() {
    for (auto& device_stream : device_streams_) {
      if (device_stream) {
        device_stream->Flush();
      }
    }
    // only clean the streams that is owned by current context
    for (auto& stream : device_streams_containers) {
      if (stream) {
        auto& allocators = stream->provider->GetAllocators();
        for (auto& alloc : allocators) {
          if (alloc->Info().alloc_type == OrtArenaAllocator) {
            auto* arena_alloc = static_cast<BFCArena*>(alloc.get());
            auto* stream_aware_alloc = static_cast<StreamAwareArena*>(arena_alloc);
            if (stream_aware_alloc) {
              stream_aware_alloc->ReleaseStreamBuffers(stream.get());
            }
          }
        }
      }
    }
  }

  void SetDeviceStream(size_t idx, std::unique_ptr<Stream> stream) {
    ORT_ENFORCE(idx < num_streams_);
    device_streams_[idx] = stream.get();
    device_streams_containers.emplace_back(std::move(stream));
  }

  void SetDeviceStream(size_t idx, Stream* stream) {
    ORT_ENFORCE(idx < num_streams_);
    device_streams_[idx] = stream;
  }

  const std::vector<Stream*>& GetStreams() {
    return device_streams_;
  }

  size_t NumStreams() { return num_streams_; }

 private:
  size_t num_streams_;
  std::vector<Stream*> device_streams_;
  std::vector<std::unique_ptr<Stream>> device_streams_containers;
};

DeviceStreamColloection::DeviceStreamColloection(size_t num_streams) : impl_(std::make_unique<DeviceStreamColloectionImpl>(num_streams)) {}

DeviceStreamColloection::~DeviceStreamColloection() {}

void DeviceStreamColloection::SetDeviceStream(size_t idx, std::unique_ptr<Stream> stream) {
  impl_->SetDeviceStream(idx, std::move(stream));
}

void DeviceStreamColloection::SetDeviceStream(size_t idx, Stream* stream) {
  impl_->SetDeviceStream(idx, stream);
}

const std::vector<Stream*>& DeviceStreamColloection::GetStreams() const {
  return impl_->GetStreams();
}

size_t DeviceStreamColloection::NumStreams() const {
  return impl_->NumStreams();
}

/*
* LIMITATION: 
* CountDownBarrier is only for scenario that the v is set 
* to the # of consumers and each consumer calls Dec() exactly once.
*/
class CountDownBarrier {
 public:
  CountDownBarrier() : v_{0} {};

  void Set(int32_t v) {
    ORT_ENFORCE(v >= 0);
    v_.store(v, std::memory_order_relaxed);
  }

  bool Dec() {
    return v_.fetch_sub(1, std::memory_order_relaxed) == 1;
  }

  int32_t Get() { return v_.load(std::memory_order_relaxed); }

 private:
  std::atomic_int_fast32_t v_;
};

// execution context that support to execute a command on stream.
// The notifications got instantiated when execution context is constructed.
// TODO: if we merge the notifications to execution frame, we might don't need this.
struct ExecutionContext {
  const SessionState* session_state;
  std::unique_ptr<ExecutionFrame> frame;
  const logging::Logger* logger;
  std::vector<std::unique_ptr<synchronize::Notification>> notifications;
  std::unique_ptr<ReleasePlan> release_plan;
  const DeviceStreamColloection& device_stream_map_;
  std::vector<CountDownBarrier> count_down_barriers_;
  CountDownBarrier remain_tasks_;
  const bool& terminate_flag_;
  Status task_status_{Status::OK()};

  ExecutionContext(const SessionState& sess_state,
                   std::vector<std::unique_ptr<LogicStream>>& logic_streams,
                   std::vector<size_t> notification_owners,
                   const std::vector<int>& feed_mlvalue_idxs,
                   const std::vector<OrtValue>& feeds, const std::vector<int>& fetch_mlvalue_idxs,
                   std::vector<OrtValue>& fetches,
                   const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                   size_t num_barriers,
                   const logging::Logger& sess_logger,
                   const DeviceStreamColloection& device_streams_map,
                   const bool& terminate_flag) : session_state(&sess_state),
                                                         logger(&sess_logger),
                                                         device_stream_map_(device_streams_map),
                                                         count_down_barriers_(num_barriers),
                                                         terminate_flag_(terminate_flag) {
    int32_t valid_streams = 0;
    //1. bind logic stream to device stream;
    for (auto& logic_stream : logic_streams) {
      if (logic_stream->commands_.size() > 0) {
        valid_streams++;
      }
    }

    auto& device_streams = device_stream_map_.GetStreams();

    for (auto i = 0; i < notification_owners.size(); ++i) {
      auto& stream = device_streams[notification_owners[i]];
      if (stream)
        notifications.push_back(std::move(stream->CreateNotification(/*TODO: calculate num of consumers*/ 0)));
      else
        notifications.push_back(nullptr);
    }

    // create frame
    frame = std::make_unique<ExecutionFrame>(feed_mlvalue_idxs, feeds, fetch_mlvalue_idxs, fetches, fetch_allocators, sess_state, &device_streams);
    // init barreris
    for (auto i = 0; i < num_barriers; ++i) {
      count_down_barriers_[i].Set(2);
    }
    remain_tasks_.Set(valid_streams);

    auto* para_exe_plan = const_cast<SessionState&>(sess_state).GetParalllelExecutionPlan();
    release_plan = para_exe_plan->GenerateReleasePlan();
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

  ~ExecutionContext() {}

  void RecycleNodeInputs(onnxruntime::NodeIndex node_index) {
    ORT_ENFORCE(frame);
    for (auto it : release_plan->node_value_map_[node_index]) {
      if (--release_plan->value_ref_counts_[it] == 0) {
        auto original_it = release_plan->reused_map_.find(it);
        if (original_it == release_plan->reused_map_.end()) {
          ORT_ENFORCE(frame->ReleaseMLValue(it).IsOK());
          LOGS(*logger, INFO) << "ort value " << it << " released";
        } else {
          if (--release_plan->value_ref_counts_[original_it->second] == 0) {
            ORT_ENFORCE(frame->ReleaseMLValue(original_it->second).IsOK());
            LOGS(*logger, INFO) << "ort value " << original_it->second << " released";
          }
        }
      }
    }
  }
};

void LogicStream::RunSince(ExecutionContext& ctx, size_t since) {
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
      status = commands_[since](ctx, continue_flag);
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

  size_t NumStreams() const { return num_logic_streams_;}

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
  std::vector<std::vector<std::string>> streams_log_;   // save up nodes per stream for logging

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
      }
      else if (create_stream_fn) {
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

  class StreamRange { //iterate between [from,to)
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
  std::unique_ptr<std::vector<NodeIndex>[]> nodes_in_stream { new std::vector<NodeIndex>[num_logic_streams_] };
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
      } 
      else {
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
          logic_streams_[i]->commands_.push_back([barrier_id](ExecutionContext& ctx, bool& continue_flag) {
            continue_flag = ctx.DecCountDownBarrier(barrier_id);
            return Status::OK();
          });
          // push a wait command if has EP registered it.
          auto wait_handle = session_state.GetStreamHandleRegistryInstance().GetWaitHandle(
              logic_streams_[notification_owners_[notfication_it->second]]->ep_->Type(),
              node->GetExecutionProviderType());
          if (wait_handle) {
            const std::string& upstream_node_name = it->Name();
            logic_streams_[i]->commands_.push_back([wait_handle, cur_stream_idx, notification_index, i, node, upstream_node_name](ExecutionContext& ctx, bool& continue_flag) {
              wait_handle(*ctx.GetDeviceStream(cur_stream_idx), *ctx.notifications[notification_index]);
              // update streams clock status
              if (ctx.GetDeviceStream(cur_stream_idx))
                ctx.GetDeviceStream(cur_stream_idx)->UpdateStreamClock(ctx.notifications[notification_index]->stream, ctx.notifications[notification_index]->timestamp);
              LOGS(*ctx.logger, INFO) << "stream " << i << " wait on " << upstream_node_name << " for " << node->Name();
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
      logic_streams_[i]->commands_.push_back([this, node, node_index, cur_stream_idx, i, &value_map](ExecutionContext& ctx, bool& continue_flag) {
        auto* p_kernel = ctx.session_state->GetKernel(node_index);
        auto* intra_tp = ctx.session_state->GetThreadPool();
        // TODO: set terminate flag from run_option
        OpKernelContextInternal kernel_ctx(*ctx.session_state, *ctx.frame, *p_kernel, *ctx.logger, ctx.terminate_flag_, ctx.GetDeviceStream(cur_stream_idx));
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

          ctx.RecycleNodeInputs(node_index);
#ifdef USE_CUDA
          nvtxRangePop();
#endif
        }
        LOGS(*ctx.logger, INFO) << "stream " << i << " complete with " << node->Name();
        continue_flag = true;
        return Status::OK();
      });
      // check if any notification generated by this node, if yes, push a activate
      auto notification_it = node_to_notification.find(node_index);
      if (notification_it != node_to_notification.end()) {
        NotificationIndex notification_index = notification_it->second;
        logic_streams_[i]->commands_.push_back([notification_index, i, node](ExecutionContext& ctx, bool& continue_flag) {
          if (ctx.notifications[notification_index])
            ctx.notifications[notification_index]->ActivateAndUpdate();
          LOGS(*ctx.logger, INFO) << "stream " << i << " send notification for " << node->Name();
          continue_flag = true;
          return Status::OK();
        });
        // notify downstreams
        logic_streams_[i]->commands_.push_back([this, notification_index](ExecutionContext& ctx, bool& continue_flag) {
          ScheduleDownstream(ctx, notification_index);
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
  // prepare the execution context, notifications got initialized.
  ExecutionContext ctx(session_state,
                       logic_streams_,
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
  ORT_RETURN_IF_ERROR(ctx.task_status_);
  ORT_RETURN_IF_ERROR(ctx.frame->GetOutputs(fetches));
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
  for (auto value_it : impl_->value_consumer_map_) {
    // skip graph inputs, ouputs and weights
    if (allocation_plan[value_it.first].alloc_kind != AllocKind::kAllocate &&
        allocation_plan[value_it.first].alloc_kind != AllocKind::kReuse) {
      continue;
    }
    for (auto node_it : value_it.second) {
      release_plan->node_value_map_[node_it].push_back(value_it.first);
    }
  }
  for (OrtValueIndex i = 0; i < allocation_plan.size(); ++i) {
    if (allocation_plan[i].reused_buffer > 0) {
      release_plan->reused_map_[i] = allocation_plan[i].reused_buffer;
    }
  }
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
            break; // only resued once
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

}  // namespace onnxruntime