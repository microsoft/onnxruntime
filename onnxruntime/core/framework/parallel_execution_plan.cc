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

void RegisterStreamCommandHanler(const SessionState& session_state) {
  auto& eps = session_state.GetExecutionProviders();
  for (auto& ep : eps) {
    ep->RegisterStreamHandlers(GetStreamHandleRegistryInstance());
  }
}

//struct ReleasePlan {
//  std::unique_ptr<std::atomic_int[]> ref_counts;
//  std::unordered_map<onnxruntime::NodeIndex, std::vector<int>> node_ref_count_map;
//};

// execution context that support to execute a command on stream.
// The notifications got instantiated when execution context is constructed.
// TODO: if we merge the notifications to execution frame, we might don't need this.
struct ExecutionContext {
  const SessionState* session_state;
  ExecutionFrame* frame;
  const logging::Logger* logger;
  std::vector<std::unique_ptr<synchronize::Notification>> notifications;
  std::unordered_map<NodeIndex, std::vector<OrtValueIndex>> release_plan;

  ExecutionContext(const SessionState& sess_state,
                   ExecutionFrame* execution_frame,
                   std::vector<Stream*> notification_owners,
                   const logging::Logger& sess_logger) : session_state(&sess_state),
                                                         frame(execution_frame),
                                                         logger(&sess_logger) {
    for (auto i = 0; i < notification_owners.size(); ++i) {
      notifications.push_back(std::move(notification_owners[i]->CreateNotification(/*TODO: calculate num of consumers*/0)));
    }
    auto* para_exe_plan = const_cast<SessionState&>(sess_state).GetParalllelExecutionPlan();
    release_plan = para_exe_plan->GenerateReleasePlan();
  }

  ~ExecutionContext() {
  }

  void RecycleNodeInputs(onnxruntime::NodeIndex node_index) {
    if (release_plan.find(node_index) != release_plan.end()) {
      for (auto value_index : release_plan[node_index]) {
        ORT_THROW_IF_ERROR(frame->ReleaseMLValue(value_index));
        LOGS(*logger, INFO) << "value " << value_index << " released";
      }
    }
  }
};

using CommandFn = std::function<void(ExecutionContext&)>;

// a logic stream to execute command.
// each command in the logic stream will be executed in FIFO
// a logic stream will be binded to multiple device stream, as the command in the same logic stream may be executed on different EPs.
// i.e., if we set concurrency level to 1, the single logic stream will be equal to our sequential execution plan, which has both cpu and gpu kernels
struct LogicStream {
  std::vector<CommandFn> commands_;

  void Run(ExecutionContext& ctx) {
    for (auto& command : commands_) {
      command(ctx);
    }
    // flush
    for (auto& device_stream : device_streams_) {
      device_stream->Flush();
    }
  }

  ~LogicStream() {}
};

struct ParallelExecutionPlanImpl {
  ParallelExecutionPlanImpl(const SessionState& session_state,
                            const ProviderStreamMap& provider_stream_map,
                            const OpStreamMap& op_stream_map);
  ~ParallelExecutionPlanImpl();

  common::Status Execute(const SessionState& session_state,
                         const std::vector<int>& feed_mlvalue_idxs,
                         const std::vector<OrtValue>& feeds,
                         const std::vector<int>& fetch_mlvalue_idxs,
                         std::vector<OrtValue>& fetches,
                         const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                         const logging::Logger& logger);

  Stream* GetComputeStreamForNode(NodeIndex index) const {
    auto it = node_to_stream_map_.find(index);
    return it == node_to_stream_map_.end() ? nullptr : it->second;
  }

  const std::vector<int>& GetRefCounts() const { return value_ref_counts_; }

  std::vector<std::unique_ptr<LogicStream>> logic_streams_;
  const SessionState& session_state_;
  int num_logic_streams_{};

  // the stream where the notificaiton got created.
  std::vector<Stream*> notification_owners_;
  std::unordered_map<NodeIndex, Stream*> node_to_stream_map_;
  std::unordered_map<size_t, Stream*> value_to_stream_map_;
  std::vector<int> value_ref_counts_;
  std::unordered_map<onnxruntime::NodeIndex, std::vector<onnxruntime::OrtValueIndex>> node_value_map_;
  std::unordered_map<onnxruntime::OrtValueIndex, onnxruntime::NodeIndex> value_node_map_;
  ProviderStreamMap provider_stream_map_;
  OpStreamMap op_stream_map_;
  std::vector<std::vector<std::string>> streams_log_;   // save up nodes per stream for logging
};

std::once_flag populate_command_handle_flag;

//todo: remove dependency on session_state

ParallelExecutionPlanImpl::ParallelExecutionPlanImpl(const SessionState& session_state,
                                                     const ProviderStreamMap& provider_stream_map,
                                                     const OpStreamMap& op_stream_map) : session_state_(session_state),
                                                                                         provider_stream_map_(provider_stream_map),
                                                                                         op_stream_map_(op_stream_map) {
  const auto& value_map = session_state_.GetOrtValueNameIdxMap();
  const auto& execution_providers = session_state_.GetExecutionProviders();
  const auto& kernel_create_info_map = session_state_.GetKernelCreateInfoMap();

  // register handle once
  std::call_once(
      populate_command_handle_flag, [](const SessionState& sess_state) { RegisterStreamCommandHanler(sess_state); }, session_state);
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
  //3. Check the nodes in each logical stream, bind it to device streams
  for (auto i = 0; i < num_logic_streams_; ++i) {
    std::set<const IExecutionProvider*> providers;
    for (auto node_index : nodes_in_stream[i]) {
      ORT_ENFORCE(node_stream_map[node_index] == i);
      auto* node = graph_viewer.GetNode(node_index);
      onnxruntime::ProviderType exec_provider_name = node->GetExecutionProviderType();
      const IExecutionProvider* ep = session_state.GetExecutionProviders().Get(exec_provider_name);
      if (providers.find(ep) == providers.end()) {
        auto create_stream_fn = GetStreamHandleRegistryInstance().GetCreateStreamFn(ep->Type());
        ORT_ENFORCE(create_stream_fn);
        logic_streams_[i]->device_streams_.emplace_back(create_stream_fn(ep));
        providers.insert(ep);
      }
      // setup node to stream map
      auto& streams = logic_streams_[node_stream_map[node_index]]->device_streams_;
      auto stream_it = std::find_if(streams.begin(),
                                    streams.end(),
                                    [&](std::unique_ptr<Stream>& stream) { return stream->provider == ep; });
      ORT_ENFORCE(stream_it != streams.end());
      node_to_stream_map_[node_index] = stream_it->get();
    }
  }
  //4. set notification owners
  notification_owners_.resize(num_notifications);
  for (auto node_index : graph_viewer.GetNodesInTopologicalOrder()) {
    auto it = node_to_notification.find(node_index);
    if (it != node_to_notification.end()) {
      // notification owned by the node who produced it.
      // use the producer's EP instance poitner as owner id
      auto* node = graph_viewer.GetNode(node_index);
      onnxruntime::ProviderType exec_provider_name = node->GetExecutionProviderType();
      const IExecutionProvider* ep = session_state.GetExecutionProviders().Get(exec_provider_name);
      auto& streams = logic_streams_[node_stream_map[node_index]]->device_streams_;
      auto stream_it = std::find_if(streams.begin(),
                                    streams.end(),
                                    [&](std::unique_ptr<Stream>& stream) { return stream->provider == ep; });
      ORT_ENFORCE(stream_it != streams.end());
      notification_owners_[it->second] = stream_it->get();
    }
  }
  //5. add commands to logic queue
  for (auto i = 0; i < num_logic_streams_; ++i) {
    for (auto node_index : nodes_in_stream[i]) {
      // check if any producer is not in current stream, if yes, create a wait
      auto* node = graph_viewer.GetNode(node_index);
      for (auto it = node->InputNodesBegin(); it != node->InputNodesEnd(); ++it) {
        if (std::find(nodes_in_stream[i].begin(), nodes_in_stream[i].end(), it->Index()) == nodes_in_stream[i].end()) {
          // find the notificaiton id
          auto notfication_it = node_to_notification.find(it->Index());
          ORT_ENFORCE(notfication_it != node_to_notification.end());
          // push a wait command
          auto wait_handle = GetStreamHandleRegistryInstance().GetWaitHandle(notification_owners_[notfication_it->second], node->GetExecutionProviderType());
          NotificationIndex notification_index = notfication_it->second;
          auto* cur_stream = node_to_stream_map_[node_index];
          const std::string& upstream_node_name = it->Name();
          logic_streams_[i]->commands_.push_back([wait_handle, cur_stream, notification_index, i, node, upstream_node_name](ExecutionContext& ctx) {
            wait_handle(*cur_stream, *ctx.notifications[notification_index]);
            LOGS(*ctx.logger, INFO) << "stream " << i << " wait on " << upstream_node_name << " for " << node->Name();
          });
        }
      }
      // push launch kernel command
      auto& streams = logic_streams_[i]->device_streams_;
      onnxruntime::ProviderType exec_provider_name = node->GetExecutionProviderType();
      const IExecutionProvider* ep = session_state.GetExecutionProviders().Get(exec_provider_name);
      auto stream_it = std::find_if(streams.begin(),
                                    streams.end(),
                                    [&](std::unique_ptr<Stream>& stream) { return stream->provider == ep; });
      ORT_ENFORCE(stream_it != streams.end());
      logic_streams_[i]->commands_.push_back([this, node, node_index, i, &value_map](ExecutionContext& ctx) {
        auto* p_kernel = ctx.session_state->GetKernel(node_index);
        auto* intra_tp = ctx.session_state->GetThreadPool();
        OpKernelContext kernel_ctx(ctx.frame, p_kernel, intra_tp, *ctx.logger);
        if (p_kernel->IsAsync()) {
          ExecutionContext* ctx_ptr = &ctx;
          ORT_ENFORCE(p_kernel->ComputeAsync(&kernel_ctx, [node_index, i, ctx_ptr]() {
                ctx_ptr->RecycleNodeInputs(node_index);
              }).IsOK(), MakeString("kernel fail!"));
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
          ORT_ENFORCE(p_kernel->Compute(&kernel_ctx).IsOK(), MakeString("kernel fail!"));

          ctx.RecycleNodeInputs(node_index);
#ifdef USE_CUDA
          nvtxRangePop();
#endif
        }
        /*
        //test indirect release - release input values of input nodes
        const auto& input_node_args = node->InputDefs();
        for (int input_index_local = 0; input_index_local < input_node_args.size(); ++input_index_local) {
          const auto* input_arg = input_node_args[input_index_local];
          OrtValueIndex input_idx_global;
          if (value_map.GetIdx(input_arg->Name(), input_idx_global).IsOK()) {
            if (value_node_map_.find(input_idx_global) != value_node_map_.end()) {
              NodeIndex owning_node = value_node_map_[input_idx_global];
              ctx.RecycleNodeInputs(owning_node);
            }
          }
        }
        if (p_kernel->IsAsync()) {
          ORT_ENFORCE(p_kernel->ComputeAsync(&kernel_ctx, []() {}).IsOK(), MakeString("kernel fail!"));
        } else {
          ORT_ENFORCE(p_kernel->Compute(&kernel_ctx).IsOK(), MakeString("kernel fail!"));
        }*/
        //test indirect release - done
        LOGS(*ctx.logger, INFO) << "stream " << i << " complete with " << node->Name();
      });
      // check if any notification generated by this node, if yes, push a activate
      auto notification_it = node_to_notification.find(node_index);
      if (notification_it != node_to_notification.end()) {
        NotificationIndex notification_index = notification_it->second;
        logic_streams_[i]->commands_.push_back([notification_index, i, node](ExecutionContext& ctx) {
          ctx.notifications[notification_index]->Activate();
          LOGS(*ctx.logger, INFO) << "stream " << i << " send notification for " << node->Name();
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
      OrtValueIndex input_idx_global;
      ORT_THROW_IF_ERROR(value_map.GetIdx(input_arg->Name(), input_idx_global));
      if (node_outputs.find(input_idx_global) != node_outputs.end()) {
        value_ref_counts_[input_idx_global]++;
        node_value_map_[node_index].push_back(input_idx_global);
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

ParallelExecutionPlanImpl::~ParallelExecutionPlanImpl() {
}

//ReleasePlan ParallelExecutionPlanImpl::GenerateReleasePlan() const {
//  ReleasePlan release_plan;
//  release_plan.ref_counts.reset(new std::atomic_int[value_ref_counts_.size()]);
//  for (int i = 0; i < value_ref_counts_.size(); ++i) {
//    release_plan.ref_counts[i] = value_ref_counts_[i];
//  }
//  release_plan.node_ref_count_map = this->node_value_map_;
//  return release_plan;
//}

common::Status ParallelExecutionPlanImpl::Execute(const SessionState& session_state, const std::vector<int>& feed_mlvalue_idxs,
                                                  const std::vector<OrtValue>& feeds, const std::vector<int>& fetch_mlvalue_idxs,
                                                  std::vector<OrtValue>& fetches,
                                                  const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                                                  const logging::Logger& logger) {
  ExecutionFrame frame(feed_mlvalue_idxs, feeds, fetch_mlvalue_idxs, fetches, fetch_allocators, session_state);
  auto* tp = session_state.GetInterOpThreadPool();
  // prepare the execution context, notifications got initialized.
  ExecutionContext execution_context(session_state, &frame, notification_owners_, logger);
  // execution_context.release_plan = GenerateReleasePlan();
  std::unique_ptr<Barrier[]> barriers{new Barrier[num_logic_streams_-1]}; // TODO: handle case when num_logic_streams_ == 0

  // WARNING: all task scheduled must be less or equal to the number 
  // of inter op threads, otherwise the execution might hang
  ORT_ENFORCE(num_logic_streams_ <= concurrency::ThreadPool::DegreeOfParallelism(tp));

  for (int i = 0; i < num_logic_streams_-1; ++i) {
    if (logic_streams_[i]->commands_.empty()) {
      barriers.get()[i].set();  // let go the stream if it is empty
    } else {
      concurrency::ThreadPool::Schedule(tp, [i, this, &barriers, &execution_context]() {
        logic_streams_[i]->Run(execution_context);
        barriers.get()[i].set();
      });
    }
  }//for

  // run last stream in main thread
  LogicStream* stream = logic_streams_[num_logic_streams_-1].get();
  stream->Run(execution_context);

  for (int i = 0; i < num_logic_streams_-1; ++i) {
    barriers[i].wait();
  }

  //TODO: we might need to flush all the stream before return the result.
  ORT_RETURN_IF_ERROR(frame.GetOutputs(fetches));
  return Status::OK();
}

ParallelExecutionPlan::ParallelExecutionPlan(const SessionState& session_state,
                                             const ProviderStreamMap& provider_stream_map,
                                             const OpStreamMap& op_stream_map) {
  impl_ = std::make_unique<ParallelExecutionPlanImpl>(session_state, provider_stream_map, op_stream_map);
}

ParallelExecutionPlan::~ParallelExecutionPlan() {
}

bool ParallelExecutionPlan::CanReuse(size_t ort_value_old, size_t ort_value_new) const {
  // only allow reuse to happen in same stream
  // TODO: enable reusing among streams under certain cases
  return impl_->value_to_stream_map_.count(ort_value_old) &&
         impl_->value_to_stream_map_.count(ort_value_new) &&
         impl_->value_to_stream_map_[ort_value_old] == impl_->value_to_stream_map_[ort_value_new];
}

common::Status ParallelExecutionPlan::Execute(const SessionState& session_state, const std::vector<int>& feed_mlvalue_idxs,
                                              const std::vector<OrtValue>& feeds, const std::vector<int>& fetch_mlvalue_idxs,
                                              std::vector<OrtValue>& fetches,
                                              const std::unordered_map<size_t, CustomAllocator>& fetch_allocators,
                                              const logging::Logger& logger) {
  return impl_->Execute(session_state, feed_mlvalue_idxs, feeds, fetch_mlvalue_idxs, fetches, fetch_allocators, logger);
}

const std::vector<AllocPlanPerValue>& ParallelExecutionPlan::GetAllocPlanPerValue() const {
  return this->allocation_plan;
}


Stream* ParallelExecutionPlan::GetComputeStreamForNode(NodeIndex index) const {
  return impl_->GetComputeStreamForNode(index);
}

const std::vector<int>& ParallelExecutionPlan::GetRefCounts() const { 
    return impl_->value_ref_counts_; 
}

std::unordered_map<NodeIndex, std::vector<OrtValueIndex>> ParallelExecutionPlan::GenerateReleasePlan() {
  std::unordered_map<NodeIndex, std::vector<OrtValueIndex>> release_plan;
  for (const auto& exe_plan : execution_plan) {
    release_plan[exe_plan.node_index] = {};
    for (int i = exe_plan.free_from_index; i < exe_plan.free_to_index; ++i) {
      release_plan[exe_plan.node_index].push_back(to_be_freed[i]);
    }
  }
  return release_plan;
}

}  // namespace onnxruntime