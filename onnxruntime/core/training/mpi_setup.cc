#include "mpi_setup.h"

namespace onnxruntime {
namespace training {
MPIContext::MPIContext(int w_rank, int l_rank, int w_size) : world_rank(w_rank), local_rank(l_rank), world_size(w_size) {}
#ifdef USE_HOROVOD
MPIContext setup_horovod() {
  using namespace horovod::common;
  // setup MPI amd horovod
  MPI_Init(0, 0);

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int* ranks = (int*)malloc(sizeof(int) * world_size);

  MPI_Allgather(&world_rank, 1, MPI_INT, ranks, 1, MPI_INT, MPI_COMM_WORLD);

  horovod_init(ranks, world_size);

  //Get local rank
  int local_rank;
  MPI_Comm shmcomm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                      MPI_INFO_NULL, &shmcomm);
  MPI_Comm_rank(shmcomm, &local_rank);

  printf("Using cuda device #%d, rank is %d, world_size %d \n",
         local_rank, world_rank, world_size);

  return MPIContext(world_rank, local_rank, world_size);
}

void shutdown_horovod() {
  horovod::common::horovod_shutdown();
  MPI_Finalize();
}
#endif
}  // namespace training
}  // namespace onnxruntime
