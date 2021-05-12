// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#ifndef SHARED_PROVIDER
#include "core/common/common.h"
#include "core/common/logging/logging.h"
#endif
#include "orttraining/core/framework/distributed_run_context.h"

#if defined(USE_MPI)
#include <mpi.h>
#endif

namespace onnxruntime {
namespace training {

#if defined(USE_MPI)
#define MPI_CHECK(condition)  \
  do {                        \
    int error = (condition);  \
    ORT_ENFORCE(              \
        error == MPI_SUCCESS, \
        "MPI Error at: ",     \
        __FILE__,             \
        ":",                  \
        __LINE__,             \
        ": ",                 \
        error);               \
  } while (0)
#endif

struct MPIGroup {
#if defined(USE_MPI)
  MPI_Group mpi_group{MPI_GROUP_EMPTY};  // MPI group
  MPI_Comm communicator{MPI_COMM_NULL};  // MPI communicator of this group
#endif
  bool is_group_initialized{false};  // Whether it's initialized
};

class MPIContext {
  // https://stackoverflow.com/questions/1008019/c-singleton-design-pattern
 public:
  static MPIContext& GetInstance();

  MPIContext(MPIContext const&) = delete;
  void operator=(MPIContext const&) = delete;

  // within ~MPIContext() we need to check for _WIN32 before calling shutdown_mpi().
  ~MPIContext();

  void AddMPIGroup(WorkerGroupType group_type, WorkerGroup& group);

  const std::vector<MPIGroup>& GetAllMPIGroups() const { return mpi_groups_; }

  const MPIGroup& GetMPIGroup(WorkerGroupType group_type) const { return mpi_groups_[group_type]; }

  int GetWorldRank() const { return world_rank_; }
  int GetLocalRank() const { return local_rank_; }
  int GetWorldSize() const { return world_size_; }
  int GetLocalSize() const { return local_size_; }

  const static int MPI_TIMEOUT_IN_SECONDS = 10;

#if defined(USE_MPI)
  // https://docs.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-best-practices
  // in case of _WIN32 we cannot call shutdown_mpi() in MPIContext destructor because of DllMain's restriction
  // shutdown_mpi shall be called specifically in user code.
  static void shutdown_mpi(bool perform_graceful_exit = true);
#endif

 private:
  MPIContext();

  // Groups containing mpi communicator for any worker group.
  std::vector<MPIGroup> mpi_groups_;
#if defined(USE_MPI)
  // Global counter for MPI groups
  int mpi_group_id_ = 0;
  void setup_mpi();
  void ReleaseComms();
#endif
  int world_rank_;
  int local_rank_;
  int world_size_;
  int local_size_;

  const logging::Logger& logger_ = logging::LoggingManager::DefaultLogger();
};

}  // namespace training
}  // namespace onnxruntime
