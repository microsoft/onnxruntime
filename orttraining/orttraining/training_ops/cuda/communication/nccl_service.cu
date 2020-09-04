// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(USE_NCCL) || defined(USE_HOROVOD)

#include "orttraining/training_ops/cuda/communication/nccl_service.cuh"
#include "core/profile/context.h"
#include <iostream>
#include <nccl.h>

namespace onnxruntime {
namespace cuda {

void NcclService::SubmitSendAndWait(void* ptr, size_t size, int peer) {
  // Wait until NCCL service is launched.
  {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this]{return is_launched_;});
  }

  auto& profile_context = profile::Context::GetInstance();
  const auto tag = profile_context.GetThreadTagOrDefault(std::this_thread::get_id());
  
  // Pointer to enqueued task.
  const NcclTask* task;

  // Submit task.
  {
    std::lock_guard<std::mutex> guard(mutex_);
    auto& profile_context = profile::Context::GetInstance();
    const auto tag = profile_context.GetThreadTagOrDefault(std::this_thread::get_id());
    task = schedule_[time_].EqueueTask(NcclTask::Type::SEND, std::vector<int>{peer}, ptr, size, tag);
  }

  // Wait for task to be finished.
  {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&]{return task->is_finished;});
  }
};

void NcclService::SubmitRecvAndWait(void* ptr, size_t size, int peer) {
  // Wait until NCCL service is launched.
  {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this]{return is_launched_;});
  }

  // Pointer to euqueued task.
  const NcclTask* task;
  {
    std::lock_guard<std::mutex> guard(mutex_);
    auto& profile_context = profile::Context::GetInstance();
    const auto tag = profile_context.GetThreadTagOrDefault(std::this_thread::get_id());
    task = schedule_[time_].EqueueTask(NcclTask::Type::RECV, std::vector<int>{peer}, ptr, size, tag);
  }

  // Wait for task to be finished.
  {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&]{return task->is_finished;});
  }
};

void NcclService::Initialize() {
  // Here we assume GPU i is assigned to local process i.
  // TODO: Create a general class to describe for computation topology and unify all similar uses.
  // Hardwoards a process can own:
  //   GPUs
  //   CPUs
  //   Other devices
  int mpi_rank;
  int mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  // Set device this NCCL communicator runs on.
  cudaSetDevice(mpi_rank);

  // Create stream to run NCCL.
  cudaStreamCreate(&stream_);

  // Get NCCL unique ID at rank 0 and broadcast it to all others.
  ncclUniqueId id;
  if (mpi_rank == 0) ncclGetUniqueId(&id);
  MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclCommInitRank(&comm_, mpi_size, id, mpi_rank);
}

void NcclService::Launch() {
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!is_planned_) {
      throw std::runtime_error("NCCL service must know its communication plan before launching.");
    }
    // The NCCL service object can only be launched once because it's a
    // singlton class.
    if (is_launched_) {
      throw std::runtime_error("NCCL service cannot be repeatedly launched.");
    }

    // Set this flag so that others will not call this again.
    is_launched_ = true;
  }

  Initialize();

  while (is_launched_) {
    // Enter critical region.
    // The state of this class cannot be modified by other threads.
    {
      std::lock_guard<std::mutex> guard(mutex_);
      // All tasks must be ready with a valid time.
      if (time_ > schedule_.size() - 1 ||
          !schedule_[time_].IsAllTasksEqueued() ||
          schedule_[time_].IsAllTasksFinished()) {
        continue;
      }

      // Start NCCL parallel communication.
      ncclGroupStart();
      for (auto& task : schedule_[time_].batch) {
        if (!task.is_enqueued) {
          throw std::runtime_error("Unscheduled task cannot be run.");
        }

        switch (task.type) {
          case NcclTask::Type::SEND:
            if (task.peers.size() != 1) {
              throw std::invalid_argument("Send can only send data to one rank.");
            }
            ncclSend(task.ptr, task.size, ncclChar, task.peers.front(), comm_, stream_);
            break;
          case NcclTask::Type::RECV:
            if (task.peers.size() != 1) {
              throw std::invalid_argument("Recv can only send data to one rank.");
            }
            ncclRecv(task.ptr, task.size, ncclChar, task.peers.front(), comm_, stream_);
            break;
          default:
            throw std::runtime_error("NCCL service currently only support ncclSend and ncclRecv.");
        }
        task.is_finished = true;
      }
      ncclGroupEnd();

      // Wait all communication to be finished.
      cudaStreamSynchronize(stream_);

      // This round of communication is done.
      // We can start waiting for the tasks to be fully scheduled.
      ++time_;
      ++total_time_;
    }
    cv_.notify_all();
  }
}

void NcclService::Reset() {
  {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this]{return is_launched_;});
  }

  for (auto& task_group : schedule_) {
    while (!task_group.IsAllTasksFinished()) {};
  }
  std::lock_guard<std::mutex> guard(mutex_);
  time_ = 0;

  // All scheduled communication tasks are done for finishing
  // gradient accumulation steps + one model update step.
  // To start next round of gradient accumulation and model update,
  // we need to reset the "done" status of all tasks in the schedule.
  for (auto& task_group : schedule_) {
    task_group.ResetAllTasks();
  }

  // 
  cv_.notify_all();
}

void NcclService::Terminate() {
  {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this]{return is_launched_ && total_time_ > 0 && time_ == 0;});
  }

  is_launched_ = false;
  cudaStreamDestroy(stream_);
  ncclCommDestroy(comm_);
}

}  // namespace cuda
}  // namespace onnxruntime

#endif