// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "activations.h"
#include "core/framework/op_kernel.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_ACTIVATION_KERNEL(x, ver, T)                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                 \
      x,                                                         \
      kOnnxDomain,                                               \
      ver,                                                       \
      T,                                                         \
      kCudaExecutionProvider,                                    \
      KernelDefBuilder()                                         \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()) \
          .MayInplace(0, 0),                                     \
      x<T>);

#define UNARY_ACTIVATION_COMPUTE(x, T)                                                                     \
  template <>                                                                                              \
  Status x<T>::ComputeInternal(OpKernelContext* context) const {                                           \
    UnaryElementwisePreparation p;                                                                         \
    UnaryElementwise::Prepare(context, &p);                                                                \
    CudaAsyncBuffer<Ctx##x> func_ctx(this, MakeFuncCtx(), 1);                                              \
    if (!std::is_same<CtxNull, Ctx##x>::value) ORT_RETURN_IF_ERROR(func_ctx.CopyToGpu());                  \
    Impl_##x<typename ToCudaType<T>::MappedType>(                                                          \
        reinterpret_cast<const typename ToCudaType<T>::MappedType*>(p.input_tensor->template Data<T>()),   \
        reinterpret_cast<typename ToCudaType<T>::MappedType*>(p.output_tensor->template MutableData<T>()), \
        func_ctx.GpuPtr(), p.output_tensor->Shape().Size());                                               \
                                                                                                           \
    return Status::OK();                                                                                   \
  }

#define UNARY_ACTIVATION_OP_TYPED(name, ver, T) \
  REGISTER_ACTIVATION_KERNEL(name, ver, T)      \
  UNARY_ACTIVATION_COMPUTE(name, T)

#define UNARY_ACTIVATION_OP_HFD(name, ver)        \
  UNARY_ACTIVATION_OP_TYPED(name, ver, MLFloat16) \
  UNARY_ACTIVATION_OP_TYPED(name, ver, float)     \
  UNARY_ACTIVATION_OP_TYPED(name, ver, double)

UNARY_ACTIVATION_OP_HFD(Affine, 1);
UNARY_ACTIVATION_OP_HFD(ParametricSoftplus, 1);
UNARY_ACTIVATION_OP_HFD(ScaledTanh, 1);


REGISTER_ACTIVATION_KERNEL(ThresholdedRelu, 1, MLFloat16)
REGISTER_ACTIVATION_KERNEL(ThresholdedRelu, 1, float)
REGISTER_ACTIVATION_KERNEL(ThresholdedRelu, 1, double)

}  //namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
