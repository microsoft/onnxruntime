// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "/bert_ort/wechi/nccl/nccl-2.7.3-1/build/include/nccl.h"
#include "orttraining/training_ops/cuda/communication/nccl_service.h"
#include "core/profile/context.h"
#include <iostream>

namespace onnxruntime {
namespace cuda {

void NcclService::SubmitSendAndWait(void* ptr, size_t size, int peer) {
  auto& profile_context = profile::Context::GetInstance();
  const auto tag = profile_context.GetThreadTagOrDefault(std::this_thread::get_id());
  
  // Pointer to enqueued task.
  const NcclTask* task;

  while (!is_launched_) {}

  // Submit task.
  {
    std::lock_guard<std::mutex> guard(mutex_);
    // std::cout << "Submit task: Batch " << tag << " send to " << peer << std::endl;
    task = schedule_[time_].EqueueTask(NcclTask::Type::SEND, std::vector<int>{peer}, ptr, size, tag);
  }

  // Wait for task to be finished.
  {
    // std::cout << "Submit task: Batch " << tag << " send to " << peer << " wait" << std::endl;
    std::unique_lock<std::mutex> lock(mutex_);

    while (!task->is_finished) {
      cv_.wait(lock);
    }

    // std::cout << "Submit task: Batch " << tag << " send to " << peer << " done" << std::endl;
  }
};

void NcclService::SubmitRecvAndWait(void* ptr, size_t size, int peer) {
  auto& profile_context = profile::Context::GetInstance();
  const auto tag = profile_context.GetThreadTagOrDefault(std::this_thread::get_id());

  while (!is_launched_) {}
  /*
  {
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    bool gdb_flag = mpi_rank == 1;
    while (gdb_flag) {
      gdb_flag = gdb_flag;
    }
  }
  */

  // Pointer to euqueued task.
  const NcclTask* task;
  {
    std::lock_guard<std::mutex> guard(mutex_);
    // std::cout << "Submit task: Batch " << tag << " recv from " << peer << std::endl;
    task = schedule_[time_].EqueueTask(NcclTask::Type::RECV, std::vector<int>{peer}, ptr, size, tag);
  }

  {
    // std::cout << "Submit task: Batch " << tag << " recv from " << peer << " wait" << std::endl;
    std::unique_lock<std::mutex> lock(mutex_);

    while (!task->is_finished) {
      cv_.wait(lock);
    }

    // std::cout << "Submit task: Batch " << tag << " recv from " << peer << " done" << std::endl;
  }
};

void NcclService::Launch() {
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!is_planned_) {
      throw;
    }
    // The NCCL service object can only be launched once because it's a
    // singlton class.
    if (is_launched_) {
      throw;
    }

    // Set this flag so that others will not call this again.
    is_launched_ = true;
  }

  std::cout << "Launch NCCL service" << std::endl;
  /*
  bool gdb_flag = true;
  while (gdb_flag) {
    gdb_flag = gdb_flag;
  }
  */
  std::cout << "Launch NCCL service done" << std::endl;

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

  std::cout << "[nccl_service.cc, Launch] enter infinite loop." << std::endl;
  while (is_launched_) {
    {
      std::lock_guard<std::mutex> guard(mutex_);
      // std::cout << "[nccl_service.cc, Launch] acquire lock." << std::endl;
      const auto time = FindNextCommunicationTime();
      if (time < 0) {
        // std::cout << "[nccl_service.cc, Launch] mpi rank: " << mpi_rank << std::endl;
        continue;
      }
      cudaStreamSynchronize(stream_);
      // std::cout << "[nccl_service.cc, Launch] found time: " << time << std::endl;
      ncclGroupStart();
      // std::cout << "[nccl_service.cc, Launch] start group call " << std::endl;
      for (auto& task : schedule_[time].batch) {
        switch (task.type) {
          case NcclTask::Type::SEND:
            if (task.peers.size() != 1) {
              throw;
            }
            // std::cout << "[nccl_service.cc, Launch] send to " << task.peers.front() << std::endl;
            ncclSend(task.ptr, task.size, ncclChar, task.peers.front(), comm_, stream_);
            break;
          case NcclTask::Type::RECV:
            if (task.peers.size() != 1) {
              throw;
            }
            // std::cout << "[nccl_service.cc, Launch] recv from " << task.peers.front() << std::endl;
            ncclRecv(task.ptr, task.size, ncclChar, task.peers.front(), comm_, stream_);
            break;
          default:
            throw;
        }
        task.is_finished = true;
      }
      ncclGroupEnd();
      ++time_;
      ++total_time_;
    }
    cudaStreamSynchronize(stream_);
    cv_.notify_all();
  }

  // std::cout << "NCCL service terminated." << std::endl;
}

void NcclService::Reset() {
  for (auto& task_group : schedule_) {
    while (!task_group.IsAllTasksFinished()) {};
  }
  std::lock_guard<std::mutex> guard(mutex_);
  time_ = 0;
  for (auto& task_group : schedule_) {
    task_group.ResetAllTasks();
  }
}

void NcclService::Terminate() {
  while (!(total_time_ > 0 && time_ == 0)) {};
  std::lock_guard<std::mutex> guard(mutex_);
  is_launched_ = false;
  ncclCommDestroy(comm_);
}

}  // namespace cuda
}  // namespace onnxruntime