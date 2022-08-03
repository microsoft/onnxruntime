/**
 * Copyright (c) 2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 **/

#include "core/providers/shared_library/provider_api.h"
#include "hailo_memcpy_op.h"

namespace onnxruntime {

HailoMemcpy::HailoMemcpy(const OpKernelInfo& info) : OpKernel(info) {}

Status HailoMemcpy::Compute(OpKernelContext* ctx) const {
    Status retval;
    MLDataType input_type_0 = ctx->InputType(0);
    if (input_type_0->IsTensorType()) {
        const auto* X = ctx->Input<Tensor>(0);
        Tensor* Y = ctx->Output(0, X->Shape());
        retval = Info().GetDataTransferManager().CopyTensor(*X, *Y, Info().GetKernelDef().ExecQueueId());
    }
#if !defined(DISABLE_SPARSE_TENSORS)
    else if (input_type_0->IsSparseTensorType()) {
        const auto* X = ctx->Input<SparseTensor>(0);
        SparseTensor* Y = ctx->OutputSparse(0, X->DenseShape());
        retval = X->Copy(Info().GetDataTransferManager(), Info().GetKernelDef().ExecQueueId(), *Y);
    }
#endif
    else {
        ORT_NOT_IMPLEMENTED("Input type not supported: ", DataTypeImpl::ToString(input_type_0));
    }

    return retval;
}

}  // namespace onnxruntime