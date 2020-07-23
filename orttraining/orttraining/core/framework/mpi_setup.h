#pragma once

#if defined(USE_NCCL) || defined(USE_HOROVOD)
#include <mpi.h>
#endif

#ifdef USE_HOROVOD
#include "orttraining/core/graph/horovod_adapters.h"
#endif

namespace onnxruntime {
namespace training {

#define MPI_CHECK(condition)                                 \
  do {                                                       \
    int error = (condition);                                 \
    ORT_ENFORCE(                                             \
        error == MPI_SUCCESS,                                \
        "MPI Error at: ",                                    \
        __FILE__,                                            \
        ":",                                                 \
        __LINE__,                                            \
        ": ",                                                \
        error);                                              \
  } while (0)

struct MPIContext {
  MPIContext(int world_rank = 0, int local_rank = 0, int world_size = 1, int local_size = 1);
  int world_rank;
  int local_rank;
  int world_size;
  int local_size;
};

#if defined(USE_NCCL) || defined(USE_HOROVOD)
MPIContext setup_mpi();
void shutdown_mpi();
#endif

}  // namespace training
}  // namespace onnxruntime
