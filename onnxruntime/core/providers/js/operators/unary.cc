// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/js/js_execution_provider.h"
#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace js {

class AbsImpl : public JsKernel {
public:
    AbsImpl(const OpKernelInfo& info) : JsKernel(info) {}

    Status Compute(OpKernelContext* context) const override {
        AllocatorPtr alloc;
        ORT_RETURN_IF_ERROR(context->GetTempSpaceCPUAllocator(&alloc));
        size_t temp_data_size = sizeof(size_t) * (1 + context->InputCount() * (3 + context->Input<Tensor>(0)->Shape().NumDimensions()));
        printf("temp data size: %zu\n", temp_data_size);
        void *p_inputs = alloc->Alloc(temp_data_size);

        //
        // type | data_ptr | dim_size | dim[0] ... dim[N-1]
        //

        Tensor* Y = context->Output(0, TensorShape(context->Input<Tensor>(0)->Shape()));
        printf("Y.data=%zu\n", (size_t)(Y->DataRaw()));

        alloc->Free(p_inputs);

        return Status::OK();
    }
};

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Abs,
    kOnnxDomain,
    1,
    14,
    kJsExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    AbsImpl);

}  // namespace js
}  // namespace onnxruntime
