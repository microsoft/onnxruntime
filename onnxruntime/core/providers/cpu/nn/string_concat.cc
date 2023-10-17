#include "string_concat.h"
#include "core/providers/cpu/element_wise_ranged_transform.h"
#include "core/providers/cpu/math/element_wise_ops.h"
#include "core/common/common.h"

namespace onnxruntime {
ONNX_CPU_OPERATOR_KERNEL(
    StringConcat,
    20,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<std::string>()),
    StringConcat);

Status StringConcat::Compute(OpKernelContext* context) const {
    const auto* X = context->Input<Tensor>(0);
    const auto* X_data = X->template Data<std::string>();
    const auto* Y = context->Input<Tensor>(1);
    const auto* Y_data = Y->template Data<std::string>();
    auto* Z = context->Output(0, X->Shape());
    auto* Z_data = Z->template MutableData<std::string>();
    const auto N = X->Shape().Size();

    for (int64_t i = 0; i < N; ++i) {
        Z_data[i] = X_data[i] + Y_data[i];
    }

    return Status::OK();
}

} // namespace onnxruntime
