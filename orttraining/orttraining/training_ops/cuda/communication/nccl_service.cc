// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(USE_CUDA) && defined(ORT_USE_NCCL) && defined(USE_NCCL_P2P)

#include "orttraining/training_ops/cuda/communication/nccl_service.h"
#include "core/common/common.h"
#include "core/profile/context.h"
#include "core/providers/cuda/cuda_check_memory.h"
#include "core/providers/cuda/cuda_common.h"
#include "orttraining/core/framework/communication/mpi/mpi_context.h"

namespace onnxruntime {
namespace cuda {

bool NcclTask::Compare(const NcclTask& other) const {
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

void NcclTask::ResetTask() {
  ptr = nullptr;
  size = 0;
  is_enqueued = false;
  is_finished = false;
}

void NcclTaskGroup::PlanTask(
    const NcclTask::Type type,
    const std::vector<int> peers) {
  batch.push_back({type, peers, nullptr, 0, false, false, ""});
};

const NcclTask* NcclTaskGroup::EqueueTask(
    const NcclTask::Type type,
    const std::vector<int> peers,
    void* ptr,
    const size_t size,
    const std::string info) {
  NcclTask scheduled_task;
  scheduled_task.type = type;
  scheduled_task.peers = peers;

  for (auto& task : batch) {
    if (!task.Compare(scheduled_task)) {
      // "scheduled_task" doesn't match "task" in task "batch" in this time slot.
      // Go checking if "scheduled_task" matches next one.
      continue;
    }

    ORT_ENFORCE(!task.is_finished, "Cannot enqueue finished NCCL P2P task again before calling ResetAllTasks in a time slot.");
    ORT_ENFORCE(!task.is_enqueued, "Cannot enqueue duplicated NCCL P2P tasks in a time slot.");

    // "scheduled_task" matches "task", so we add the task details for launching NCCL call.
    task.ptr = ptr;
    task.size = size;
    task.is_enqueued = true;
    task.info = info;
    return &task;
  }

  return nullptr;
};

bool NcclTaskGroup::IsAllTasksEqueued() const {
  return std::all_of(
      batch.begin(), batch.end(), [&](const NcclTask& task) {
        return task.is_enqueued;
      });
};

bool NcclTaskGroup::IsAllTasksFinished() const {
  return std::all_of(
      batch.begin(), batch.end(), [&](const NcclTask& task) {
        return task.is_finished;
      });
};

void NcclTaskGroup::ResetAllTasks() {
  for (auto& task : batch) {
    task.ptr = nullptr;
    task.size = 0;
    task.is_enqueued = false;
    task.is_finished = false;
  }
};

void NcclService::PlanStart() {
  ORT_ENFORCE(!is_planned_, "Communication plan cannot be changed after calling PlanEnd.");
};

void NcclService::PlanEnd() {
  is_planned_ = true;
};

void NcclService::PlanNewGroupStart() {
  group_status_.push_back(true);
  schedule_.push_back(NcclTaskGroup());
};

void NcclService::PlanNewGroupEnd() {
  group_status_.back() = false;
};

void NcclService::PlanSend(const int dst) {
  ORT_ENFORCE(group_status_.back(), "Last communication group can not be changed after call PlanEndNewGroup.");

  schedule_.back().PlanTask(NcclTask::Type::SEND, {dst});
};

void NcclService::PlanRecv(const int src) {
  ORT_ENFORCE(group_status_.back(), "Last communication group can not be changed after call PlanEndNewGroup.");
  schedule_.back().PlanTask(NcclTask::Type::RECV, {src});
};

void NcclService::WaitForLaunch() {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this] { return is_running_; });
}

std::ostream& operator<<(std::ostream& stream, const NcclTaskGroup& task_group) {
  for (int i = 0; static_cast<size_t>(i) < task_group.batch.size(); ++i) {
    std::string line = "  ";
    auto& task = task_group.batch[i];
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
    stream << line;
  }
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const NcclService& service) {
  for (int i = 0; static_cast<size_t>(i) < service.schedule_.size(); ++i) {
    stream << "NCCL operations at time " << i << std::endl;
    stream << service.schedule_[i];
  }
  return stream;
}

int NcclService::FindNextCommunicationTime() const {
  for (int i = 0; static_cast<size_t>(i) < schedule_.size(); ++i) {
    if (schedule_[i].IsAllTasksEqueued() && !schedule_[i].IsAllTasksFinished()) {
      return i;
    }
  }
  return -1;
};

void NcclService::SubmitSendAndWait(void* ptr, size_t size, int peer) {
  // Wait until NCCL service is launched.
  WaitForLaunch();
  // Pointer to enqueued task.
  const NcclTask* task;

  // Submit task.
  {
    std::lock_guard<std::mutex> guard(mutex_);
#ifdef ENABLE_NVTX_PROFILE
    auto& profile_context = profile::Context::GetInstance();
    const std::string tag = profile_context.GetThreadTagOrDefault(std::this_thread::get_id());
#else
    const std::string tag = "";
#endif
    task = schedule_[time_].EqueueTask(NcclTask::Type::SEND, std::vector<int>{peer}, ptr, size, tag);

    ORT_ENFORCE(task, "Unplanned NCCL Send encountered.");
  }

  // Wait for task to be finished.
  {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&] { return task->is_finished; });
  }
};

void NcclService::SubmitRecvAndWait(void* ptr, size_t size, int peer) {
  // Wait until NCCL service is launched.
  WaitForLaunch();

  // Pointer to euqueued task.
  const NcclTask* task;
  {
    std::lock_guard<std::mutex> guard(mutex_);
#ifdef ENABLE_NVTX_PROFILE
    auto& profile_context = profile::Context::GetInstance();
    const std::string tag = profile_context.GetThreadTagOrDefault(std::this_thread::get_id());
#else
    const std::string tag = "";
#endif
    task = schedule_[time_].EqueueTask(NcclTask::Type::RECV, std::vector<int>{peer}, ptr, size, tag);
    ORT_ENFORCE(task, "Unplanned NCCL Send encountered.");
  }

  // Wait for task to be finished.
  {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&] { return task->is_finished; });
  }
};

void NcclService::Initialize() {
  // Here we assume GPU i is assigned to local process i.
  // TODO: Create a general class to describe for computation topology and unify all similar uses.
  // Hardware a process can own:
  //   GPUs
  //   CPUs
  //   Other devices

  const int mpi_rank = onnxruntime::training::MPIContext::GetInstance().GetWorldRank();
  const int mpi_local_rank = onnxruntime::training::MPIContext::GetInstance().GetLocalRank();
  const int mpi_size = onnxruntime::training::MPIContext::GetInstance().GetWorldSize();

  // Set device this NCCL communicator runs on.
  CUDA_CALL(cudaSetDevice(mpi_local_rank));

  // Create communication stream.
  CUDA_CALL(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));

  // Get NCCL unique ID at rank 0 and broadcast it to all others.
  ncclUniqueId id;
  if (mpi_rank == 0) NCCL_CALL(ncclGetUniqueId(&id));
  MPI_CHECK(MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
  NCCL_CALL(ncclCommInitRank(&comm_, mpi_size, id, mpi_rank));
}

void NcclService::Launch() {
  worker_ = std::thread([this]() {
    {
      std::lock_guard<std::mutex> guard(mutex_);
      ORT_ENFORCE(is_planned_, "NCCL service must know its communication plan before launching.");
      // The NCCL service object can only be launched once because it's a
      // singlton class.
      ORT_ENFORCE(!is_running_, "NCCL service cannot be repeatedly launched.");

      // Set this flag so that others will not call this again.
      is_running_ = true;
      cv_.notify_all();
    }

    Initialize();

    while (is_running_) {
      // Enter critical region of performing concurrent NCCL P2P operations (for example, Send's and Recv's).
      // The state of this class cannot be modified by other threads.
      {
        std::lock_guard<std::mutex> guard(mutex_);
        // All tasks must be ready with a valid time.
        if (schedule_.empty() ||
            time_ > schedule_.size() - 1 ||
            !schedule_[time_].IsAllTasksEqueued() ||
            schedule_[time_].IsAllTasksFinished()) {
          continue;
        }

        // Start NCCL parallel communication.
        NCCL_CALL(ncclGroupStart());
        for (auto& task : schedule_[time_].batch) {
          ORT_ENFORCE(task.is_enqueued,
                      "Unscheduled task cannot be run.",
                      " Use PlanTask to schedule tasks before executing the graph.");
#ifndef NDEBUG
          CheckIfMemoryOnCurrentGpuDevice(task.ptr);
#endif
          switch (task.type) {
            case NcclTask::Type::SEND:
              ORT_ENFORCE(task.peers.size() == 1, "Send can only send data to one rank.");
              NCCL_CALL(ncclSend(task.ptr, task.size, ncclChar, task.peers.front(), comm_, stream_));
              break;
            case NcclTask::Type::RECV:
              ORT_ENFORCE(task.peers.size() == 1, "Recv can only send data to one rank.");
              NCCL_CALL(ncclRecv(task.ptr, task.size, ncclChar, task.peers.front(), comm_, stream_));
              break;
            default:
              ORT_NOT_IMPLEMENTED("NCCL service currently only support ncclSend and ncclRecv.");
          }
          task.is_finished = true;
        }
        NCCL_CALL(ncclGroupEnd());

        // Make sure all NCCL computation are done.
        // Since the Submit*andWait are blocked by the following "cv_.notify_all()",
        // all NCCL Send and Recv called above are all finished before Submit*andWait returning.
        // Thus, CUDA operations after Send and Recv won't be inserted by other threads
        // when we call NCCL Send's and Recv's.
        CUDA_CALL(cudaStreamSynchronize(stream_));

        // This round of communication is done.
        // We can start waiting for the tasks to be fully scheduled.
        ++time_;
        ++total_time_;
      }
      cv_.notify_all();
    }
  });
}

void NcclService::Reset() {
  WaitForLaunch();
  {
    std::unique_lock<std::mutex> lock(mutex_);

    // We can only reset after all planned tasks are done,
    // so wait for unfinished tasks here.
    cv_.wait(lock, [this] {
      bool is_all_tasks_finished = true;
      for (auto& task_group : schedule_) {
        if (task_group.IsAllTasksFinished()) {
          continue;
        }
        is_all_tasks_finished = false;
      };
      return is_all_tasks_finished;
    });
  }

  {
    std::lock_guard<std::mutex> guard(mutex_);
    time_ = 0;

    // All scheduled communication tasks are done for finishing
    // gradient accumulation steps + one model update step.
    // To start next round of gradient accumulation and model update,
    // we need to reset the "done" status of all tasks in the schedule.
    for (auto& task_group : schedule_) {
      task_group.ResetAllTasks();
    }

    cv_.notify_all();
  }
}

void NcclService::Terminate() {
  WaitForLaunch();
  {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return schedule_.empty() || total_time_ > 0 && time_ == 0; });
  }

  CUDA_CALL(cudaStreamDestroy(stream_));

  is_running_ = false;
  is_planned_ = false;
  time_ = 0;
  total_time_ = 0;

  group_status_.clear();
  schedule_.clear();
  worker_.join();
  NCCL_CALL(ncclCommDestroy(comm_));
}

}  // namespace cuda
}  // namespace onnxruntime

#endif