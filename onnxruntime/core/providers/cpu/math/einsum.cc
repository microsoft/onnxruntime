#include "einsum.h"
#include "einsum_utils.h"

namespace onnxruntime {

// Creative credit: Implementation heavily influenced by PyTorch implementation at the time of writing

ONNX_CPU_OPERATOR_KERNEL(
    Einsum,
    12,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllNumericTensorTypes()),
    Einsum);

Status Einsum::Compute(OpKernelContext* context) const {
  return EinsumTypedProcessor<float>(context, equation_);
}

}  // namespace onnxruntime
