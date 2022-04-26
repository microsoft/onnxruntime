// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/spin_pause.h"
#include "core/framework/parallel_execution_plan.h"
#include "core/framework/session_state.h"
#include "core/framework/execution_frame.h"
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
struct Notification {
  void notify() { ready_.store(true); };
  void wait() {
    while (!ready_.load()) {
      onnxruntime::concurrency::SpinPause();
    }
  };
  std::atomic_bool ready_{};
};

using NotificationIndex = size_t;
using kernel_invoke_fn = std::function<void(NodeIndex)>;
using notification_wait_fn = std::function<void(NotificationIndex)>;
using notification_notify_fn = std::function<void(NotificationIndex)>;

struct LogicStream {
 
  std::vector<NodeIndex> nodes_;
  std::unordered_map<NodeIndex, NotificationIndex> downstream_notifications_;
  std::unordered_map<NodeIndex, std::vector<NotificationIndex>> upstream_notifications_;

  //void Reset() { // must be call before Run all all LogicStreams in ParallelExecutionPlanImpl
  //  for (auto& iter : downstream_notifications_) {
  //    iter.second->Reset(); // reset all notifications for current thread
  //  }
  //}

  void Run(const kernel_invoke_fn& invoke_fn, const notification_wait_fn& wait_fn, const notification_notify_fn& notify_fn) {
    for (auto node : nodes_) { //FIFO
      auto upstream_iter = upstream_notifications_.find(node); //todo: measure the impact of .find(...)
      if (upstream_iter != upstream_notifications_.end()) {
        for (auto notification : upstream_iter->second) {
          //std::cout << node << " wait for " << notification << std::endl;
          wait_fn(notification);
        }
      }
      invoke_fn(node);
      //std::cout << node << " run" << std::endl;
      auto downstream_iter = downstream_notifications_.find(node);
      if (downstream_iter != downstream_notifications_.end()) {
        notify_fn(downstream_iter->second);
        //std::cout << node << " send" << downstream_iter->second << std::endl;
      }
    }
  }

  NotificationIndex AddNotificationOn(NodeIndex node, NotificationIndex notification) {
    //auto node_iter = std::find(nodes_.begin(), nodes_.end(), node);
    //todo: deal with iter == nodes.end()
    auto notification_iter = downstream_notifications_.find(node);
    if (notification_iter == downstream_notifications_.end()) {
      downstream_notifications_[node] = notification;
    }
    return downstream_notifications_[node];
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
  size_t num_notifications_{};
};

//todo: remove dependency on session_state
ParallelExecutionPlanImpl::ParallelExecutionPlanImpl(const SessionState& session_state,
                                                     int num_logic_streams) : session_state_(session_state), num_logic_streams_(num_logic_streams) {
  std::vector<std::vector<std::string>> streams_stdout;
  for (int i = 0; i < num_logic_streams_; ++i) {
    logic_streams_.push_back(std::make_unique<LogicStream>());
    streams_stdout.push_back(std::vector<std::string>{});
  }
  int stream_iter = 0;
  const auto& graph_viewer = session_state_.GetGraphViewer();
  const auto& value_map = session_state_.GetOrtValueNameIdxMap();
  std::unordered_map<NodeIndex, int> node_to_stream;
  std::unordered_map<int, NodeIndex> output_to_node;
  for (auto node_index : graph_viewer.GetNodesInTopologicalOrder()) {
    const auto* node = graph_viewer.GetNode(node_index);
    //auto& stream = logic_streams_[stream_iter % num_logic_streams_];
    std::vector<NotificationIndex> upstream_notifications;
    for (auto input : node->InputDefs()) {
      int idx = -1;
      ORT_THROW_IF_ERROR(value_map.GetIdx(input->Name(), idx));
      NodeIndex upstream_node = output_to_node[idx];
      int upstream_node_stream = node_to_stream[upstream_node];
      // todo: devise a better allocation algo, with benchmarks
      if (stream_iter != upstream_node_stream) {
        auto downstream_notification = logic_streams_[upstream_node_stream]->AddNotificationOn(upstream_node, num_notifications_);
        if (downstream_notification == num_notifications_) {
          num_notifications_++;
        }
        upstream_notifications.push_back(downstream_notification);
        auto upstream_node_name = graph_viewer.GetNode(upstream_node)->OpType();
        for (auto iter = streams_stdout[upstream_node_stream].begin(); iter < streams_stdout[upstream_node_stream].end(); ++iter) {
          if (*iter == upstream_node_name) {
            if ((next(iter) == streams_stdout[upstream_node_stream].end() ||
                 *next(iter) != upstream_node_name + "_send_notification")) {
              streams_stdout[upstream_node_stream].insert(iter + 1, upstream_node_name + "_send_notification");
              streams_stdout[stream_iter].push_back("wait_for_" + upstream_node_name);
            }
            break;
          }
        }
      }
    }
    logic_streams_[stream_iter]->nodes_.push_back(node_index);
    logic_streams_[stream_iter]->upstream_notifications_[node_index] = std::move(upstream_notifications);
    streams_stdout[stream_iter].push_back(node->OpType());
    node_to_stream[node_index] = stream_iter;
    stream_iter = (stream_iter + 1) % num_logic_streams_;
    for (auto output : node->OutputDefs()) {
      int idx = -1;
      ORT_THROW_IF_ERROR(value_map.GetIdx(output->Name(), idx));
      output_to_node[idx] = node_index;
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
  auto* intra_tp = session_state.GetThreadPool();
  std::unique_ptr<Barrier[]> barriers{new Barrier[num_logic_streams_-1]}; //todo: handle case when num_logic_streams_ == 0

  std::function<void(NodeIndex)> run_fn = [&] (NodeIndex node_index) {
    auto* kernel = session_state.GetKernel(node_index);
    OpKernelContext ctx(&frame, kernel, intra_tp, logger);
    ORT_ENFORCE(kernel->Compute(&ctx).IsOK(), MakeString("kernel fail!"));
  };

  std::unique_ptr<Notification[]> notifications{new Notification[num_notifications_]}; 

  std::function<void(NotificationIndex)> wait_fn = [&](NotificationIndex notification) {
    notifications[notification].wait();
  };

  std::function<void(NotificationIndex)> notify_fn = [&](NotificationIndex notification) {
    notifications[notification].notify();
  };

  for (int i = 0; i < num_logic_streams_-1; ++i) {
    LogicStream* stream = logic_streams_[i].get();
    Barrier* barrier = &barriers.get()[i];
    concurrency::ThreadPool::Schedule(tp, [&]() {
      stream->Run(run_fn, wait_fn, notify_fn);
      barrier->set();
    });
  }//for

  // run last stream in main thread
  LogicStream* stream = logic_streams_[num_logic_streams_-1].get();
  stream->Run(run_fn, wait_fn, notify_fn);

  for (int i = 0; i < num_logic_streams_-1; ++i) {
    barriers[i].wait();
  }

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