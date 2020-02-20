// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "activations.h"
#include "core/framework/op_kernel.h"

using namespace onnxruntime::hip;

namespace onnxruntime {
namespace contrib {
namespace hip {

#define REGISTER_ACTIVATION_KERNEL(x, ver, domain, T)            \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                 \
      x,                                                         \
      domain,                                                    \
      ver,                                                       \
      T,                                                         \
      kHipExecutionProvider,                                    \
      KernelDefBuilder()                                         \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()) \
          .MayInplace(0, 0),                                     \
      x<T>);

#define UNARY_ACTIVATION_COMPUTE(x, T)                                                                     \
  template <>                                                                                              \
  Status x<T>::ComputeInternal(OpKernelContext* context) const {                                           \
    UnaryElementwisePreparation p;                                                                         \
    UnaryElementwise::Prepare(context, &p);                                                                \
    HipAsyncBuffer<Ctx##x> func_ctx(this, MakeFuncCtx(), 1);                                              \
    if (!std::is_same<CtxNull, Ctx##x>::value) ORT_RETURN_IF_ERROR(func_ctx.CopyToGpu());                  \
    Impl_##x<typename ToHipType<T>::MappedType>(                                                          \
        reinterpret_cast<const typename ToHipType<T>::MappedType*>(p.input_tensor->template Data<T>()),   \
        reinterpret_cast<typename ToHipType<T>::MappedType*>(p.output_tensor->template MutableData<T>()), \
        func_ctx.GpuPtr(), p.output_tensor->Shape().Size());                                               \
                                                                                                           \
    return Status::OK();                                                                                   \
  }

#define UNARY_ACTIVATION_OP_TYPED(name, ver, domain, T) \
  REGISTER_ACTIVATION_KERNEL(name, ver, domain, T)      \
  UNARY_ACTIVATION_COMPUTE(name, T)

#define UNARY_ACTIVATION_OP_HFD(name, ver, domain)        \
  UNARY_ACTIVATION_OP_TYPED(name, ver, domain, MLFloat16) \
  UNARY_ACTIVATION_OP_TYPED(name, ver, domain, float)     \
  UNARY_ACTIVATION_OP_TYPED(name, ver, domain, double)

// UNARY_ACTIVATION_OP_HFD(Affine, 1, kOnnxDomain);
// UNARY_ACTIVATION_OP_HFD(ParametricSoftplus, 1, kOnnxDomain);
// UNARY_ACTIVATION_OP_HFD(ScaledTanh, 1, kOnnxDomain);
UNARY_ACTIVATION_OP_HFD(Gelu, 1, kMSDomain);

// REGISTER_ACTIVATION_KERNEL(ThresholdedRelu, 1, kOnnxDomain, MLFloat16)
// REGISTER_ACTIVATION_KERNEL(ThresholdedRelu, 1, kOnnxDomain, float)
// REGISTER_ACTIVATION_KERNEL(ThresholdedRelu, 1, kOnnxDomain, double)

}  //namespace hip
}  // namespace contrib
}  // namespace onnxruntime
