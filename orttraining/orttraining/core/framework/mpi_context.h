#pragma once

#if defined(USE_NCCL)
#include <mpi.h>
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

class MPIContext {
  // https://stackoverflow.com/questions/1008019/c-singleton-design-pattern
  public:
    static const MPIContext& GetInstance();

    MPIContext(MPIContext const&) = delete;
    void operator=(MPIContext const&) = delete;
    
    ~MPIContext();

    int GetWorldRank() const { return world_rank; }
    int GetLocalRank() const { return local_rank; }
    int GetWorldSize() const { return world_size; }
    int GetLocalSize() const { return local_size; }

  private:
    MPIContext();
#if defined(USE_NCCL)
    void setup_mpi();
    void shutdown_mpi();
#endif

    int world_rank;
    int local_rank;
    int world_size;
    int local_size;

};

}  // namespace training
}  // namespace onnxruntime
