#include "core/common/common.h"
#include <mpi.h>
#include <nccl.h>

namespace onnxruntime {
namespace distributed {

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

class ProcessGroup {
 public:
  ProcessGroup(size_t group_id, size_t size, size_t rank);
  virtual ~ProcessGroup();
  ncclComm_t GetNcclCommunicator() { return nccl_comm_; }

 private:
  ncclComm_t nccl_comm_{nullptr};
};

}  // namespace distributed
}  // namespace onnxruntime
