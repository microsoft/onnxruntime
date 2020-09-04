// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(USE_NCCL) || defined(USE_HOROVOD)

#pragma once
#include <condition_variable>
#include <list>
#include <mutex>
#include <map>
#include <vector>
#include <iostream>
#include <string>

#include <nccl.h>
#include <mpi.h>

namespace onnxruntime {
namespace cuda {

struct NcclTask final {
  // Attributes for communication operator.
  enum class Type {SEND, RECV, ALLREDUCE};
  // Operator to perform.
  Type type;
  // For Send, this field is destination device's ID.
  // For Recv, this field is source device's ID.
  std::vector<int> peers;

  // Attributes for memory location.
  // GPU memory pointer.
  void* ptr;
  // Number of bytes to send/recv.
  size_t size;

  // Scheduler flag.
  bool is_enqueued;
  bool is_finished;

  // Debug information
  std::string info;

  // Return true if the two operations are the same.
  bool Compare(const NcclTask& other) const {
    if (type != other.type) {
      return false;
    }
    if (peers.size() != other.peers.size()) {
      return false;
    }
    for (size_t i = 0; i < peers.size(); ++i) {
      if (peers[i] != other.peers[i]) {
        return false;
      }
    }

    return true;
  }

  void ResetTask() {
    ptr = nullptr;
    size = 0;
    is_enqueued = false;
    is_finished = false;
  }
};

class NcclTaskGroup final {
public:
  void PlanTask(const NcclTask::Type type, const std::vector<int> peers) {
    batch.push_back({type, peers, nullptr, 0, false, false, ""});
  };

  // Fill in task's details.
  const NcclTask* EqueueTask(const NcclTask::Type type, const std::vector<int> peers, void* ptr, const size_t size, const std::string info) {
    NcclTask scheduled_task;
    scheduled_task.type = type;
    scheduled_task.peers = peers;

    for (auto& task : batch) {
      if (!task.Compare(scheduled_task)) {
        continue;
      }
      if (task.is_finished || task.is_enqueued) {
        throw;
      }

      task.ptr = ptr;
      task.size = size;
      task.is_enqueued = true;
      task.info = info;
      return &task;
    }

    return nullptr;
  };

  bool IsAllTasksEqueued() {
    for (auto& task : batch) {
      if (!task.is_enqueued) {
        return false;
      }
    }
    return true;
  };

  bool IsAllTasksFinished() {
    for (auto& task : batch) {
      if (!task.is_finished) {
        return false;
      }
    }
    return true;
  };

  void ResetAllTasks () {
    for (auto& task : batch) {
      task.ptr = nullptr;
      task.size = 0;
      task.is_enqueued = false;
      task.is_finished = false;
    }
  };

  void Show(const std::string& prefix) const {
    for (int i = 0; static_cast<size_t>(i) < batch.size(); ++i) {
      std::string line = prefix;
      auto& task = batch[i];
      if (task.type == NcclTask::Type::SEND) {
        line += "Send, ";
      } else if (task.type == NcclTask::Type::RECV) {
        line += "Recv, ";
      }

      for (int j = 0; static_cast<size_t>(j) < task.peers.size(); ++j) {
        line += std::to_string(task.peers[j]);
        if (static_cast<size_t>(j) != task.peers.size() - 1) {
          line += ", ";
        } else {
          line += "\n";
        }
      }
      std::cout << line;
    }
  }

  std::vector<NcclTask> batch;
};

class NcclService final {
 public:
  static NcclService& GetInstance() {
    static NcclService instance_;
    return instance_;
  };

  void SubmitSendAndWait(void* buffer, size_t count, int peer);

  void SubmitRecvAndWait(void* buffer, size_t count, int peer);

  void StartPlan() {
    if (is_planned_) {
      throw;
    }
  };

  void PlanStartNewGroup() {
    group_status_.push_back(true); 
    schedule_.push_back(NcclTaskGroup());
  };
  void PlanSend(const int dst) {
    if (!group_status_.back()) {
      throw;
    }
    
    schedule_.back().PlanTask(NcclTask::Type::SEND, {dst});
  };
  void PlanRecv(const int src) {
    if (!group_status_.back()) {
      throw;
    }
    schedule_.back().PlanTask(NcclTask::Type::RECV, {src});
  };
  void PlanEndNewGroup() {
    group_status_.back() = false;
  }
  void EndPlan() {
    is_planned_ = true;
  }

  void Launch();

  void Reset();

  void Terminate();

  void Show() const {
    for (int i = 0; static_cast<size_t>(i) < schedule_.size(); ++i) {
      std::cout << "Time " << i << std::endl;
      schedule_[i].Show("  ");
    }
  }

 private:
  NcclService() = default;
  ~NcclService() = default;
  NcclService(const NcclService&) = delete;
  NcclService& operator=(const NcclService&) = delete;
  void Initialize();

  int FindNextCommunicationTime() {
    for (int i = 0; static_cast<size_t>(i) < schedule_.size(); ++i) {
      if (schedule_[i].IsAllTasksEqueued() && !schedule_[i].IsAllTasksFinished()) {
        return i;
      }
    }
    return -1;
  }

  // Mutex to gurantee thread-safe access to this class.
  std::mutex mutex_;
  // Conditional variable used to wait for the mutex.
  std::condition_variable cv_;

  // Stream for running NCCL.
  cudaStream_t stream_;
  ncclComm_t comm_;

  bool is_launched_;
  bool is_planned_;
  // Pipeline stage.
  size_t rank_;

  size_t time_;
  size_t total_time_;
  std::vector<bool> group_status_;
  std::vector<NcclTaskGroup> schedule_;
};

}  // namespace cuda
}  // namespace onnxruntime

#endif