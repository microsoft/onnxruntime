#include "core/providers/cpu/collectives/horovod_kernels.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    HorovodAllReduceOp,
    kOnnxDomain,
    1,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    HorovodAllReduceOp);
}  // namespace cuda
}  // namespace onnxruntime