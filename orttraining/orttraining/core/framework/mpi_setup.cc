#include <stdio.h>
#include <stdlib.h>

#include "mpi_setup.h"

namespace onnxruntime {
namespace training {
MPIContext::MPIContext(int w_rank, int l_rank, int w_size, int l_size) : world_rank(w_rank), local_rank(l_rank), world_size(w_size), local_size(l_size) {}

#if defined(USE_NCCL) || defined(USE_HOROVOD)
MPIContext setup_mpi() {
  // setup MPI amd horovod
  int is_mpi_initialized = 0;
  MPI_Initialized(&is_mpi_initialized);
  if (!is_mpi_initialized) {
    int mpi_threads_provided = 0;
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &mpi_threads_provided);
  }

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int* ranks = (int*)malloc(sizeof(int) * world_size);

  MPI_Allgather(&world_rank, 1, MPI_INT, ranks, 1, MPI_INT, MPI_COMM_WORLD);

#ifdef USE_HOROVOD
  using namespace horovod::common;
  horovod_init(ranks, world_size);
#endif

  //Get local rank and size
  int local_rank;
  int local_size;

  MPI_Comm shmcomm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                      MPI_INFO_NULL, &shmcomm);
  MPI_Comm_rank(shmcomm, &local_rank);
  MPI_Comm_size(shmcomm, &local_size);

  // Get version
  int len;
  char version[MPI_MAX_LIBRARY_VERSION_STRING];
  MPI_Get_library_version(version, &len);

  printf("Using cuda local_rank: %d, world_rank: %d, world_size: %d, local_size: %d\n(version: %s)\n",
         local_rank, world_rank, world_size, local_size, version);
  return MPIContext(world_rank, local_rank, world_size, local_size);
}

void shutdown_mpi() {
#ifdef USE_HOROVOD
  horovod::common::horovod_shutdown();
#endif

  int is_mpi_initialized = 0;
  MPI_Initialized(&is_mpi_initialized);
  if (!is_mpi_initialized)
    return;

  int is_mpi_finalized = 0;
  MPI_Finalized(&is_mpi_finalized);
  if (!is_mpi_finalized) {
    MPI_Finalize();
  }
}
#endif

}  // namespace training
}  // namespace onnxruntime
