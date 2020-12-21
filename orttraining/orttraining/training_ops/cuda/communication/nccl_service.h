// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(USE_CUDA) && defined(ORT_USE_NCCL) && defined(USE_NCCL_P2P)

#pragma once
#include <condition_variable>
#include <list>
#include <mutex>
#include <map>
#include <vector>
#include <iostream>
#include <string>
#include <thread>

#include <nccl.h>

#include <mpi.h>

namespace onnxruntime {
namespace cuda {

struct NcclTask final {
  // Attributes for communication operator.
  enum class Type { SEND,
                    RECV,
                    ALLREDUCE };
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
  bool Compare(const NcclTask& other) const;

  // Clear runtime information.
  void ResetTask();
};

// A collection of independent communication operations.
struct NcclTaskGroup final {
  // Schedule a communication operation in this group.
  // We don't know the pointer to the actual data and other runtime information yet;
  // runtime information is filled by calling EqunueTask(...).
  void PlanTask(const NcclTask::Type type, const std::vector<int> peers);
  // Fill in task's details.
  const NcclTask* EqueueTask(
      const NcclTask::Type type,
      const std::vector<int> peers,
      void* ptr,
      const size_t size,
      const std::string info);
  bool IsAllTasksEqueued() const;
  bool IsAllTasksFinished() const;
  void ResetAllTasks();
  friend std::ostream& operator<<(std::ostream& stream, const NcclTaskGroup& task_group);
  std::vector<NcclTask> batch;
};

// The use of this class has two stages. First, the user needs to plan the communication operators.
// Second, when running a model, the user should submit tasks following the communication plan.
// Function names begin with "Plan" are used for creating communication plan. Function names begin
// with "Submit" asks this class to run the submitted task. Communication usually does not happen
// immediately after submitting a task. The actual communication time is decided by this class based on
// the communication plan.
//
// Below is an example of planning tasks. Notice that the communication operations in the same group are
// called in random order, so those operations cannot have mutual dependency.
//
//   auto& nccl_service = cuda::NcclService::GetInstance();
//
//   nccl_service.PlanStart();         // Signal the begin of communication planning.
//
//   nccl_service.PlanStartNewGroup(); // Create new time slot.
//   nccl_service.PlanSend(0);
//   nccl_service.PlanRecv(1);
//   nccl_service.PlanEndNewGroup();   // Mark the end of the first time slot.
//
//   nccl_service.PlanStartNewGroup(); // Create the second time slot.
//   nccl_service.PlanSend(1);
//   nccl_service.PlanRecv(0);
//   nccl_service.PlanEndNewGroup();   // Mark the end of the second time slot.
//
//   nccl_service.EndPlan();           // Signal the end of communication planning.
class NcclService final {
 public:
  // Get the singleton of this class.
  static NcclService& GetInstance() {
    static NcclService instance_;
    return instance_;
  };

  // Planning APIs. They are not thread-safe.

  // Mark the start of entire plan.
  void PlanStart();
  // Mark the end of entire plan.
  void PlanEnd();
  // Mark the begin of a new communication group. It uses the latest time slot.
  // Operations in a group can happen in random order.
  void PlanNewGroupStart();
  // Mark the end of the current communication group.
  void PlanNewGroupEnd();
  // Add Send to the current communication group.
  void PlanSend(const int dst);
  // Add Recv to the current communication group.
  void PlanRecv(const int src);

  // Runtime APIs. They are thread-safe.

  // Launch NCCL service. It's an infinite loop which repeatedly calls corresponding NCCL
  // when planned operators (e.g., Send and Recv) arrive.
  void Launch();
  // Submit a Send request with needed information such as tensor's address and number bytes to send.
  void SubmitSendAndWait(void* buffer, size_t count, int peer);
  // Submit a Recv request with needed information such as tensor's address and number bytes to recv.
  void SubmitRecvAndWait(void* buffer, size_t count, int peer);
  // Reset communication plan's status so that we can reuse the same communication plan for multiple
  // model update steps.
  void Reset();
  // Terminate NCCL service.
  void Terminate();

  // Print debug string.
  friend std::ostream& operator<<(std::ostream& stream, const NcclService& service);

 private:
  NcclService() = default;
  ~NcclService() = default;
  NcclService(const NcclService&) = delete;
  NcclService& operator=(const NcclService&) = delete;
  // Initialization for running NCCL service.
  void Initialize();
  // Most member functions should start with a call to this function because
  // they are valid only after NCCL service is launched.
  void WaitForLaunch();
  // Search the next unfinished communication group to work on.
  int FindNextCommunicationTime() const;

  // Mutex to gurantee thread-safe access to this class.
  std::mutex mutex_;
  // Conditional variable used to wait for the mutex.
  std::condition_variable cv_;

  // Stream for running NCCL.
  cudaStream_t stream_;
  ncclComm_t comm_;

  // Indicates if NCCL service launched.
  bool is_running_;
  // Indicates if NCCL service has a plan, which must be true when calling Launch(...).
  bool is_planned_;
  // Pipeline stage.
  size_t rank_;

  size_t time_;
  size_t total_time_;

  // group_status_[t] indicates if the t-th group's plan is done. Once group_status_[t] is
  // set to false, we can add communication operations to that group.
  std::vector<bool> group_status_;
  // schedule_[t] communication group at time t. Communication group at time t-1 must be
  // finished before working on the group at time t. In other words, communication groups
  // are stored in their actual time order.
  std::vector<NcclTaskGroup> schedule_;
  // Thread to asynchronously run Launc(...).
  std::thread worker_;
};

}  // namespace cuda
}  // namespace onnxruntime

#endif