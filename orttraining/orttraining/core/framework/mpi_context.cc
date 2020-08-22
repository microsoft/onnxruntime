#include <stdio.h>
#include <stdlib.h>

#include "mpi_context.h"

namespace onnxruntime {
namespace training {
MPIContext::MPIContext() :
world_rank_(0),
local_rank_(0),
world_size_(1),
local_size_(1)
{
#if defined(USE_NCCL)
  setup_mpi();
#endif
}

MPIContext::~MPIContext() {
#if defined(USE_NCCL)
#ifndef _WIN32
  shutdown_mpi();
#endif
#endif
}

const MPIContext& MPIContext::GetInstance() {
  static MPIContext context;
  return context;
}

#if defined(USE_NCCL)
void MPIContext::setup_mpi() {
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
  this->world_rank_ = world_rank;
  this->local_rank_ = local_rank;
  this->world_size_ = world_size;
  this->local_size_ = local_size;
}

void MPIContext::shutdown_mpi() {
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
