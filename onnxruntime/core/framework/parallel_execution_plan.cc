// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/spin_pause.h"
#include "core/framework/parallel_execution_plan.h"
#include "core/framework/session_state.h"
#include "core/framework/execution_frame.h"
#include "core/graph/constants.h"
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

//todo: make it general across EPs
struct CPUNotification {
  void notify() { ready_.store(true); };
  void wait() {
    while (!ready_.load()) {
      onnxruntime::concurrency::SpinPause();
    }
  };
  std::atomic_bool ready_{};
};

using NotificationIndex = size_t;
// this opaque handle could be anything the target device generated.
// it could be a cuda event, or a cpu notification implementation
using NotificationHandle = void*;
// it can be either a cuda stream, or even nullptr for device doesn't have stream support like cpu.
using StreamHandle = void*;

// a stream abstraction which hold an opaque handle, and a reference to which EP instance this stream belong to.
// it need to be EP instance as we might have different stream on different EP with same type.
// i.e. different cuda stream on different GPU.
struct Stream {
  StreamHandle handle;
  const IExecutionProvider* provider;

  Stream::Stream(StreamHandle h, const IExecutionProvider* p) : handle(h), provider(p) {}
};

// similiar as Stream
struct Notification {
  NotificationHandle handle;
  const IExecutionProvider* provider;
};

// the definition for the handle for stream commands
// EP can register the handle to the executor.
// in the POC, just use primitive function pointer
// TODO: use a better way to dispatch handles.
using WaitNotificationFn = std::function<void(Notification&)>;
using NotifyNotificationFn = std::function<void(Notification&)>;
using CreateNotificationFn = std::function<void*(const Stream&)>;
using ReleaseNotificationFn = std::function<void(void*)>;
using KernelLaunchFn = std::function<void(const OpKernel*, OpKernelContext*)>;

// a simple registry which hold the handles EP registered.
class StreamCommandHandleRegistry {
public:
  CreateNotificationFn GetCreateNotificationFn(Stream* stream) {
   auto it = create_notification_map_.find(stream->provider->Type());
   return it == create_notification_map_.end() ? nullptr : it->second;
  }

  ReleaseNotificationFn GetReleaseNotificationFn(Stream* stream) {
    auto it = release_notification_map_.find(stream->provider->Type());
    return it == release_notification_map_.end() ? nullptr : it->second;
  }

  KernelLaunchFn GetKernelLaunchFn(Stream* stream) {
    auto it = kernel_launch_map_.find(stream->provider->Type());
    return it == kernel_launch_map_.end() ? nullptr : it->second;
  }

  // Wait is a little special as we need to consider the source stream the notification generated, and the stream we are waiting.
  // i.e., for an cuda event what notify the memory copy, it could be wait on a CPU stream, or on another cuda stream.
  WaitNotificationFn GetWaitHandle(Stream* notification_owner_stream, const std::string& executor_ep_type) {
    auto it = notification_wait_map_.find(GetWaitKey(notification_owner_stream->provider->Type(), executor_ep_type));
    return it == notification_wait_map_.end() ? nullptr : it->second;
  }

  NotifyNotificationFn GetNotifyHandle(Stream* notification_owner_stream) {
    auto it = notification_notify_map_.find(notification_owner_stream->provider->Type());
    return it == notification_notify_map_.end() ? nullptr : it->second;
  }

  static StreamCommandHandleRegistry& GetInstance() {
    static StreamCommandHandleRegistry instance_;
    return instance_;
  }

  void RegisterCreateNotificationFn(const std::string& ep_type, CreateNotificationFn fn) {
    create_notification_map_.insert({ep_type, fn});
  }
  void RegisterReleaseNotificationFn(const std::string& ep_type, ReleaseNotificationFn fn) {
    release_notification_map_.insert({ep_type, fn});
  }

  void RegisterLaunchKenrelFn(const std::string& ep_type, KernelLaunchFn fn) {
    kernel_launch_map_.insert({ep_type, fn});
  }
  void RegisterWaitFn(const std::string& notification_ep_type, const std::string& ep_type, WaitNotificationFn fn) {
    notification_wait_map_.insert({GetWaitKey(notification_ep_type, ep_type), fn});
  }
  void RegisterNotifyFn(const std::string& notification_ep_type, NotifyNotificationFn fn) {
    notification_notify_map_.insert({notification_ep_type, fn});
  }


 private:
  StreamCommandHandleRegistry() = default;

  inline std::string GetWaitKey(const std::string& notificaiton_ep_type, const std::string& executor_ep_type) {
    return std::string(notificaiton_ep_type) + ":" + executor_ep_type;
  }

  std::unordered_map<std::string, CreateNotificationFn> create_notification_map_;
  std::unordered_map<std::string, ReleaseNotificationFn> release_notification_map_;
  std::unordered_map<std::string, KernelLaunchFn> kernel_launch_map_;
  std::unordered_map<std::string, WaitNotificationFn> notification_wait_map_;
  std::unordered_map<std::string, NotifyNotificationFn> notification_notify_map_;
};

// CPU Stream command handles
void LaunchCPUKernel(const OpKernel* kernel, OpKernelContext* ctx) {
  ORT_ENFORCE(kernel->Compute(ctx).IsOK(), MakeString("kernel fail!"));
}

void WaitCPUNotification(Notification& notification) {
  ORT_ENFORCE(notification.provider->Type() == kCpuExecutionProvider);
  static_cast<CPUNotification*>(notification.handle)->wait();
}

void NotifyCPUNotification(Notification& notification) {
  ORT_ENFORCE(notification.provider->Type() == kCpuExecutionProvider);
  static_cast<CPUNotification*>(notification.handle)->notify();
}

void* CreateCPUNotification(const Stream&) {
  return static_cast<void*>(new CPUNotification());
}

void ReleaseCPUNotification(void* handle) {
  delete static_cast<CPUNotification*>(handle);
}

void RegisterStreamCommandHanler() {
  StreamCommandHandleRegistry::GetInstance().RegisterLaunchKenrelFn(kCpuExecutionProvider, LaunchCPUKernel);
  StreamCommandHandleRegistry::GetInstance().RegisterWaitFn(kCpuExecutionProvider, kCpuExecutionProvider, WaitCPUNotification);
  StreamCommandHandleRegistry::GetInstance().RegisterNotifyFn(kCpuExecutionProvider, NotifyCPUNotification);
  StreamCommandHandleRegistry::GetInstance().RegisterCreateNotificationFn(kCpuExecutionProvider, CreateCPUNotification);
  StreamCommandHandleRegistry::GetInstance().RegisterReleaseNotificationFn(kCpuExecutionProvider, ReleaseCPUNotification);
}

// execution context that support to execute a command on stream.
// The notifications got instantiated when execution context is constructed.
// TODO: if we merge the notifications to execution frame, we might don't need this.
struct ExecutionContext {
  const SessionState* session_state;
  ExecutionFrame* frame;
  const logging::Logger* logger;
  std::unique_ptr<Notification[]> notifications;
  std::vector<ReleaseNotificationFn> notification_release_fns;

  ExecutionContext(const SessionState& sess_state,
      ExecutionFrame* execution_frame,
      std::vector<Stream*> notification_owners,
      const logging::Logger& sess_logger) : session_state(&sess_state), 
                                            frame(execution_frame), 
                                            logger(&sess_logger),
                                            notifications(new Notification[notification_owners.size()]){
    for (auto i = 0; i < notification_owners.size(); ++i) {
      auto create_notification_fn = StreamCommandHandleRegistry::GetInstance().GetCreateNotificationFn(notification_owners[i]);
      notifications[i].handle = create_notification_fn(*notification_owners[i]);
      notifications[i].provider = notification_owners[i]->provider;
      notification_release_fns.push_back(
        StreamCommandHandleRegistry::GetInstance().GetReleaseNotificationFn(notification_owners[i])
      );
    }
  }

  ~ExecutionContext() {
    for (auto i = 0; i < notification_release_fns.size(); ++i) {
      notification_release_fns[i](notifications[i].handle);
    }
  }
};

using CommandFn = std::function<void(ExecutionContext&)>;

// a logic stream to execute command.
// each command in the logic stream will be executed in FIFO
// a logic stream will be binded to multiple device stream, as the command in the same logic stream may be executed on different EPs.
// i.e., if we set concurrency level to 1, the single logic stream will be equal to our sequential execution plan, which has both cpu and gpu kernels
struct LogicStream {
 
  std::vector<std::unique_ptr<Stream>> device_streams_;
  std::vector<CommandFn> commands_;
  
  void Run(ExecutionContext& ctx) {
    for (auto& command : commands_) {
      command(ctx);
    }
  }

};

struct LogicStream;

struct ParallelExecutionPlanImpl {
  ParallelExecutionPlanImpl(const SessionState& session_state, int num_logic_streams);
  ~ParallelExecutionPlanImpl();
  common::Status Execute(const SessionState& session_state,
                         const std::vector<int>& feed_mlvalue_idxs,
                         const std::vector<OrtValue>& feeds,
                         const std::vector<int>& fetch_mlvalue_idxs,
                         std::vector<OrtValue>& fetches,
                         const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                         const logging::Logger& logger);
  std::vector<std::unique_ptr<LogicStream>> logic_streams_;
  const SessionState& session_state_;
  int num_logic_streams_{};
  // the stream where the notificaiton got created.
  std::vector<Stream*> notification_owners_;
};

std::once_flag populate_command_handle_flag;

//todo: remove dependency on session_state
ParallelExecutionPlanImpl::ParallelExecutionPlanImpl(const SessionState& session_state,
                                                     int num_logic_streams) : session_state_(session_state), num_logic_streams_(num_logic_streams) {
  // register handle once
  std::call_once(populate_command_handle_flag, []() { RegisterStreamCommandHanler(); });
  // instantiate logic streams
  std::vector<std::vector<std::string>> streams_stdout;
  for (int i = 0; i < num_logic_streams_; ++i) {
    logic_streams_.push_back(std::make_unique<LogicStream>());
    streams_stdout.push_back(std::vector<std::string>{});
  }
  
  const auto& graph_viewer = session_state_.GetGraphViewer();
  
  //1. partition the nodes into streams
  std::unique_ptr<std::vector<NodeIndex>[]> nodes_in_stream { new std::vector<NodeIndex>[num_logic_streams_] };
  std::unique_ptr<size_t[]> node_stream_map{new size_t[graph_viewer.MaxNodeIndex()]};
  // todo: devise a better allocation algo, with benchmarks
  int stream_iter = 0;
  for (auto node_index : graph_viewer.GetNodesInTopologicalOrder()) {
    nodes_in_stream[stream_iter].push_back(node_index);
    streams_stdout[stream_iter].push_back(graph_viewer.GetNode(node_index)->OpType());
    node_stream_map[node_index] = stream_iter;
    stream_iter = (stream_iter + 1) % num_logic_streams_;
  }
  //2. for each node, if any of its consumer partitioned to another stream, generate a notification
  size_t num_notifications=0;
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
      auto* node = graph_viewer.GetNode(node_index);
      onnxruntime::ProviderType exec_provider_name = node->GetExecutionProviderType();
      const IExecutionProvider* ep = session_state.GetExecutionProviders().Get(exec_provider_name);
      if (providers.find(ep) == providers.end()) {
        //TODO: invoke the stream creation method for different EP.
        //For prototype, hardcode to CPU
        ORT_ENFORCE(node->GetExecutionProviderType() == kCpuExecutionProvider);
        logic_streams_[i]->device_streams_.emplace_back(std::make_unique<Stream>(nullptr, ep));
        providers.insert(ep);
      }
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
          auto wait_handle = StreamCommandHandleRegistry::GetInstance().GetWaitHandle(notification_owners_[notfication_it->second], node->GetExecutionProviderType());
          NotificationIndex notification_index = notfication_it->second;
          logic_streams_[i]->commands_.push_back([wait_handle, notification_index](ExecutionContext& ctx) {
            wait_handle(ctx.notifications[notification_index]);
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
      auto kernel_launch_handle = StreamCommandHandleRegistry::GetInstance().GetKernelLaunchFn(stream_it->get());
      logic_streams_[i]->commands_.push_back([kernel_launch_handle, node_index](ExecutionContext& ctx) {
        auto* p_kernel = ctx.session_state->GetKernel(node_index);
        auto* intra_tp = ctx.session_state->GetThreadPool();
        OpKernelContext kernel_ctx(ctx.frame, p_kernel, intra_tp, *ctx.logger);
        kernel_launch_handle(p_kernel, &kernel_ctx);
      });
      // check if any notification generated by this node, if yes, push a notify
      auto notification_it = node_to_notification.find(node_index);
      if (notification_it != node_to_notification.end()) {
        auto notify_handle = StreamCommandHandleRegistry::GetInstance().GetNotifyHandle(stream_it->get());
        NotificationIndex notification_index = notification_it->second;
        logic_streams_[i]->commands_.push_back([notify_handle, notification_index](ExecutionContext& ctx) {
          notify_handle(ctx.notifications[notification_index]);
        });
      }
    }
  }

  std::function<std::string(const std::string&)> shape_output = [&](const std::string& s) {
    if (s.size() < 10) {
      return "node_" + s + "_computation";
    } else {
      return s;
    }
  };

  std::cout << logic_streams_.size() << " logic stream created" << std::endl;
  for (int i = 0; i < logic_streams_.size(); ++i) {
    std::cout << " -------- logic stream " << i;
  }
  std::cout << std::endl;
  for (int i = 0;; ++i) {
    bool has_out = false;
    for (int j = 0; j < streams_stdout.size(); ++j) {
      if (i < streams_stdout[j].size()) {
        has_out = true;
        std::cout << "      " << shape_output(streams_stdout[j][i]);
      } else {
        std::cout << "               ";
      }
    }
    std::cout << std::endl;
    if (!has_out) break;
  }
}

ParallelExecutionPlanImpl::~ParallelExecutionPlanImpl() {

}

common::Status ParallelExecutionPlanImpl::Execute(const SessionState& session_state, const std::vector<int>& feed_mlvalue_idxs,
                                                  const std::vector<OrtValue>& feeds, const std::vector<int>& fetch_mlvalue_idxs,
                                                  std::vector<OrtValue>& fetches,
                                                  const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                                                  const logging::Logger& logger) {
  ExecutionFrame frame(feed_mlvalue_idxs, feeds, fetch_mlvalue_idxs, fetches, fetch_allocators, session_state);
  auto* tp = session_state.GetInterOpThreadPool();
  // prepare the execution context, notifications got initialized.
  ExecutionContext execution_context(session_state, &frame, notification_owners_, logger);
  std::unique_ptr<Barrier[]> barriers{new Barrier[num_logic_streams_-1]}; //todo: handle case when num_logic_streams_ == 0

  for (int i = 0; i < num_logic_streams_-1; ++i) {
    LogicStream* stream = logic_streams_[i].get();
    Barrier* barrier = &barriers.get()[i];
    concurrency::ThreadPool::Schedule(tp, [&]() {
      stream->Run(execution_context);
      barrier->set();
    });
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

ParallelExecutionPlan::ParallelExecutionPlan(const SessionState& session_state, int num_logic_streams) {
  impl_ = std::make_unique<ParallelExecutionPlanImpl>(session_state, num_logic_streams);
}

ParallelExecutionPlan::~ParallelExecutionPlan() {
}

common::Status ParallelExecutionPlan::Execute(const SessionState& session_state, const std::vector<int>& feed_mlvalue_idxs,
                                              const std::vector<OrtValue>& feeds, const std::vector<int>& fetch_mlvalue_idxs,
                                              std::vector<OrtValue>& fetches,
                                              const std::unordered_map<size_t, CustomAllocator>& fetch_allocators,
                                              const logging::Logger& logger) {
  return impl_->Execute(session_state, feed_mlvalue_idxs, feeds, fetch_mlvalue_idxs, fetches, fetch_allocators, logger);
}

}  // namespace onnxruntime