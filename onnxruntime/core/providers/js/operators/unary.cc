// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/js/js_execution_provider.h"
#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace js {

class AbsImpl : public JsKernel {
public:
    AbsImpl(const OpKernelInfo& info) : JsKernel(info) {
        JSEP_INIT_KERNEL(Abs);
    }
};


// class kJsExecutionProvider_Abs_kOnnxDomain_ver1_14;
// template <> KernelCreateInfo BuildKernelCreateInfo<kJsExecutionProvider_Abs_kOnnxDomain_ver1_14>() {
//     return KernelCreateInfo(
//         KernelDefBuilder()
//         .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
//         .SetName("Abs")
//         .SetDomain(kOnnxDomain)
//         .SinceVersion(1, 14)
//         .Provider(kJsExecutionProvider).Build(),
//         static_cast<KernelCreatePtrFn>(
//             [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
//                 out = std::make_unique<AbsImpl>(info);
//                 return Status::OK();
//             })
//         );
// }

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
