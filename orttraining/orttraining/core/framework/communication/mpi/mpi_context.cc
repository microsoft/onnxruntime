// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define SHARED_PROVIDER_TODO 0

#include "orttraining/core/framework/communication/mpi/mpi_context.h"
#ifndef _WIN32
#include <chrono>
#include <thread>
#include <pthread.h>
#endif

namespace onnxruntime {
namespace training {

const int MPIContext::MPI_TIMEOUT_IN_SECONDS;

MPIContext::~MPIContext() {
#ifdef USE_MPI
#ifndef _WIN32
  // Assume ungraceful shutdown.
  std::atomic<bool> perform_graceful_exit{false};
  auto release_func_executor_thread = std::thread([this, &perform_graceful_exit]() {
    ReleaseComms();
    perform_graceful_exit = true;
  });
  // Wait MPI_TIMEOUT_IN_SECONDS seconds and check the flag again, if it's still false,
  // that means some process has crashed, not responding or unable to complete pending communications
  // due to unknown reasons, we then proceed to an ungraceful exit.
  std::this_thread::sleep_for(std::chrono::seconds(MPIContext::MPI_TIMEOUT_IN_SECONDS));
  if (!perform_graceful_exit) {
#if SHARED_PROVIDER_TODO
    LOGS(logger_, INFO) << "MPI is not able to gracefully shut down. Aborting MPI.";
#endif
    // Request to cancel the thread since it's not responsive.
    pthread_t native_handle = release_func_executor_thread.native_handle();
    pthread_cancel(native_handle);
  }
  if (release_func_executor_thread.joinable()) {
    release_func_executor_thread.join();
  }
  shutdown_mpi(perform_graceful_exit);
#else
  ReleaseComms();
#endif  // _WIN32
#endif  // USE_MPI
}

MPIContext& MPIContext::GetInstance() {
  static MPIContext context;
  return context;
}

// Default constructor of MPIContext only creates global parallel group i.e. MPI_WORLD
// and nodel local parallel group.
MPIContext::MPIContext() : world_rank_(0),
                           local_rank_(0),
                           world_size_(1),
                           local_size_(1) {
#ifdef USE_MPI
  // setup MPI
  int is_mpi_initialized = 0;
  MPI_CHECK(MPI_Initialized(&is_mpi_initialized));
  if (!is_mpi_initialized) {
    int mpi_threads_provided = 0;
    MPI_CHECK(MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &mpi_threads_provided));
  }

  int world_size;
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

  int world_rank;
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));

  int* ranks = (int*)malloc(sizeof(int) * world_size);

  MPI_Allgather(&world_rank, 1, MPI_INT, ranks, 1, MPI_INT, MPI_COMM_WORLD);

  //Get local rank and size
  int local_rank;
  int local_size;

  MPI_Comm shmcomm;
  MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                                MPI_INFO_NULL, &shmcomm));
  MPI_CHECK(MPI_Comm_rank(shmcomm, &local_rank));
  MPI_CHECK(MPI_Comm_size(shmcomm, &local_size));

  // Get version
  int len;
  char version[MPI_MAX_LIBRARY_VERSION_STRING];
  MPI_Get_library_version(version, &len);

#if SHARED_PROVIDER_TODO
  LOGS(logger_, INFO) << "MPI context initialized. World size: " << world_size
                      << ". World rank: " << world_rank
                      << ". Local size: " << local_size
                      << ". Local rank: " << local_rank;
#endif

  mpi_groups_.resize(WorkerGroupType::WorkerGroupTypeCount);
  // Create global parallel group
  // We duplicate MPI_WORLD_COMM here to avoid freeing MPI_WORLD_COMM at the end which is illegal.
  MPI_Comm global_comm;
  MPI_CHECK(MPI_Comm_dup(MPI_COMM_WORLD, &global_comm));
  MPI_Group mpi_group;
  MPI_CHECK(MPI_Comm_group(global_comm, &mpi_group));

  MPIGroup global_group = {mpi_group, global_comm, true};

  mpi_groups_[WorkerGroupType::GlobalParallel] = global_group;

  // Create node local parallel group
  MPI_Group mpi_node_local_group;
  MPI_CHECK(MPI_Comm_group(shmcomm, &mpi_node_local_group));

  MPIGroup node_local_group = {mpi_node_local_group, shmcomm, true};

  mpi_groups_[WorkerGroupType::NodeLocalDataParallel] = node_local_group;
  this->world_rank_ = world_rank;
  this->local_rank_ = local_rank;
  this->world_size_ = world_size;
  this->local_size_ = local_size;
#endif
}

#ifdef USE_MPI
void MPIContext::ReleaseComms() {
  // If MPI is finalized, return right away.
  int is_mpi_finalized = 0;
  MPI_Finalized(&is_mpi_finalized);
  if (is_mpi_finalized)
    return;
  for (auto group : mpi_groups_) {
    if (group.is_group_initialized) {
      MPI_CHECK(MPI_Group_free(&group.mpi_group));
#ifndef _WIN32
      MPI_CHECK(MPI_Comm_disconnect(&group.communicator));
#else
      MPI_CHECK(MPI_Comm_free(&group.communicator));
#endif
    }
  }
}
#endif

void MPIContext::AddMPIGroup(WorkerGroupType group_type, WorkerGroup& group) {
#ifdef USE_MPI
  auto group_name = DistributedRunContext::GetInstance().GetWorkerGroupName(group_type);
  if (this->mpi_groups_[group_type].is_group_initialized) {
#if SHARED_PROVIDER_TODO
    LOGS(logger_, INFO) << "Group " << group_name << " already exists. Re-initializing with different ranks.";
#endif
    MPI_CHECK(MPI_Group_free(&this->mpi_groups_[group_type].mpi_group));
    MPI_CHECK(MPI_Comm_free(&this->mpi_groups_[group_type].communicator));
  }

  // Create MPI new group
  MPI_CHECK(MPI_Group_incl(this->mpi_groups_[WorkerGroupType::GlobalParallel].mpi_group,
                           group.ranks.size(),
                           group.ranks.data(),
                           &this->mpi_groups_[group_type].mpi_group));

  // Create new MPI communicator for this group
  MPI_CHECK(MPI_Comm_create_group(this->mpi_groups_[WorkerGroupType::GlobalParallel].communicator,
                                  this->mpi_groups_[group_type].mpi_group,
                                  ++mpi_group_id_,
                                  &(this->mpi_groups_[group_type].communicator)));
  ORT_ENFORCE(this->mpi_groups_[group_type].communicator != MPI_COMM_NULL,
              "Failed to add new MPI group for worker group: ",
              DistributedRunContext::GetInstance().GetWorkerGroupName(group_type));
#else
  ORT_THROW("ORT must be built with MPI to add ", DistributedRunContext::GetInstance().GetWorkerGroupName(group_type), " with group id: ", group.group_id);
#endif
}

#ifdef USE_MPI
void MPIContext::shutdown_mpi(bool perform_graceful_exit /*default=true*/) {
  int is_mpi_initialized = 0;
  MPI_CHECK(MPI_Initialized(&is_mpi_initialized));
  if (!is_mpi_initialized)
    return;

  int is_mpi_finalized = 0;
  MPI_CHECK(MPI_Finalized(&is_mpi_finalized));
  if (!is_mpi_finalized) {
    if (perform_graceful_exit) {
      MPI_Finalize();
    } else {
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
  }
}
#endif

}  // namespace training
}  // namespace onnxruntime
