#include "process_group.h"

namespace onnxruntime {
namespace distributed {
ProcessGroup::ProcessGroup(size_t group_id, size_t size, size_t rank) {
  int is_mpi_initialized = 0;
  MPI_Initialized(&is_mpi_initialized);
  if (!is_mpi_initialized) {
    int mpi_threads_provided = 0;
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &mpi_threads_provided);
  }

  // Create new MPI communicator
  MPI_Comm mpi_comm;
  MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, group_id, rank, &mpi_comm));
  ORT_ENFORCE(mpi_comm != MPI_COMM_NULL, "MPI communicator creation failed.");

  // Create new NCCL communicator
  ncclUniqueId nccl_id;
  if (rank == 0) {
    if (ncclGetUniqueId(&nccl_id) != ncclSuccess)
      ORT_THROW("nccl generate unique id failed.");
  }
  MPI_CHECK(MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, mpi_comm));
  if (ncclCommInitRank(&nccl_comm_, size, nccl_id, rank) != ncclSuccess) {
    ORT_THROW("create nccl comm failed");
  }

  // Clean up
  MPI_CHECK(MPI_Comm_free(&mpi_comm));
}

ProcessGroup::~ProcessGroup() {
  if (nccl_comm_ != nullptr) {
    ncclCommDestroy(nccl_comm_);
  }
#ifdef USE_MPI
  int is_mpi_finalized = 0;
  MPI_Finalized(&is_mpi_finalized);
  if (!is_mpi_finalized) {
    MPI_Finalize();
  }
#endif
}

}  // namespace distributed
}  // namespace onnxruntime
