#pragma once

#ifdef USE_MPI
#include <mpi.h>
#endif

#ifdef USE_HOROVOD
#include "orttraining/core/graph/horovod_adapters.h"
#endif

namespace onnxruntime {
namespace training {

struct MPIContext {
  MPIContext(int world_rank = 0, int local_rank = 0, int world_size = 1, int local_size = 1);
  int world_rank;
  int local_rank;
  int world_size;
  int local_size;
};

#ifdef USE_MPI
MPIContext setup_mpi();
void shutdown_mpi();
#endif

}  // namespace training
}  // namespace onnxruntime
