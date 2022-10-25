// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cann/nn/conv.h"

using onnxruntime::common::Status;
namespace onnxruntime {
namespace cann {

using ConvPadVector = ConvAttributes::ConvPadVector;

template <typename T>
Status Conv<T>::Prepare(OpKernelContext* ctx, CannPreparation& prepare) const {
  const auto* X = ctx->Input<Tensor>(0);
  const auto* W = ctx->Input<Tensor>(1);
  const Tensor* B = ctx->Input<Tensor>(2);
  const int64_t N = X->Shape()[0];
  const int64_t M = W->Shape()[0];

  ORT_RETURN_IF_ERROR(conv_attrs_.ValidateInputShape(X, W));

  TensorShapeVector kernel_shape;
  ORT_RETURN_IF_ERROR(conv_attrs_.ComputeKernelShape(W->Shape(), kernel_shape));

  ConvPadVector pads(conv_attrs_.pads);
  if (pads.empty()) {
    pads.resize(kernel_shape.size() * 2, 0);
  }
  TensorShapeVector dilations(conv_attrs_.dilations);
  if (dilations.empty()) {
    dilations.resize(kernel_shape.size(), 1);
  }
  TensorShapeVector strides(conv_attrs_.strides);
  if (strides.empty()) {
    strides.resize(kernel_shape.size(), 1);
  }

  TensorShapeVector Y_dims({N, M});
  TensorShape input_shape = X->Shape().Slice(2);
  ORT_RETURN_IF_ERROR(conv_attrs_.InferPadsAndOutputShape(input_shape, kernel_shape, strides, dilations, pads, Y_dims));
  Tensor* Y = ctx->Output(0, TensorShape(Y_dims));

  if (strides.size() < 4) {
    strides.insert(strides.begin(), {1, 1});
  }
  if (dilations.size() < 4) {
    dilations.insert(dilations.begin(), {1, 1});
  }

  const aclDataType aclType = getACLType<T>();
  aclFormat format = ACL_FORMAT_NCHW;

  CANN_RETURN_IF_ERROR(aclopSetAttrInt(prepare.opAttr_, "auto_pad", static_cast<int64_t>(conv_attrs_.auto_pad)));
  CANN_RETURN_IF_ERROR(aclopSetAttrInt(prepare.opAttr_, "group", conv_attrs_.group));
  CANN_RETURN_IF_ERROR(aclopSetAttrListInt(prepare.opAttr_, "dilations", dilations.size(), dilations.data()));
  CANN_RETURN_IF_ERROR(aclopSetAttrListInt(prepare.opAttr_, "strides", strides.size(), strides.data()));
  CANN_RETURN_IF_ERROR(aclopSetAttrListInt(prepare.opAttr_, "pads", pads.size(), pads.data()));

  ORT_TRY {
    CANN_PREPARE_INPUTDESC(prepare, aclType, X->Shape().NumDimensions(), X->Shape().GetDims().data(), format);
    CANN_PREPARE_INPUTDESC(prepare, aclType, W->Shape().NumDimensions(), W->Shape().GetDims().data(), format);
    if (ctx->InputCount() >= 3) {
      CANN_PREPARE_INPUTDESC(prepare, aclType, B->Shape().NumDimensions(), B->Shape().GetDims().data(), format);
    } else {
      CANN_PREPARE_INPUTDESC(prepare, ACL_DT_UNDEFINED, 0, nullptr, ACL_FORMAT_UNDEFINED);
    }
    CANN_PREPARE_OUTPUTDESC(prepare, aclType, Y->Shape().NumDimensions(), Y->Shape().GetDims().data(), format);

    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<T*>(X->template Data<T>()), X->SizeInBytes());
    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<T*>(W->template Data<T>()), W->SizeInBytes());
    if (ctx->InputCount() >= 3) {
      CANN_PREPARE_INPUTBUFFER(prepare, const_cast<T*>(B->template Data<T>()), B->SizeInBytes());
    } else {
      CANN_PREPARE_INPUTBUFFER(prepare, nullptr, 0);
    }
    CANN_PREPARE_OUTPUTBUFFER(prepare, Y->template MutableData<T>(), Y->SizeInBytes());
  }
  ORT_CATCH(const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
  }

  return Status::OK();
}

#define REGISTER_CONV_TYPED_COMPUTE(T)                                         \
  template <>                                                                  \
  Status Conv<T>::ComputeInternal(OpKernelContext* context) const {            \
    CannPreparation prepare;                                                   \
    ORT_RETURN_IF_ERROR(Prepare(context, prepare));                            \
    CANN_RETURN_IF_ERROR(aclopCompileAndExecute("Conv2D",                      \
                                                prepare.inputDesc_.size(),     \
                                                prepare.inputDesc_.data(),     \
                                                prepare.inputBuffers_.data(),  \
                                                prepare.outputDesc_.size(),    \
                                                prepare.outputDesc_.data(),    \
                                                prepare.outputBuffers_.data(), \
                                                prepare.opAttr_,               \
                                                ACL_ENGINE_SYS,                \
                                                ACL_COMPILE_SYS,               \
                                                NULL,                          \
                                                Stream()));                    \
    return Status::OK();                                                       \
  }

#define REGISTER_CONV_TYPED_KERNEL(ver, T)                                                 \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      Conv,                                                                                \
      kOnnxDomain,                                                                         \
      ver,                                                                                 \
      T,                                                                                   \
      kCannExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Conv<T>);

#define REGISTER_CONV_VERSIONED_TYPED_KERNEL(startver, endver, T)                          \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                 \
      Conv,                                                                                \
      kOnnxDomain,                                                                         \
      startver,                                                                            \
      endver,                                                                              \
      T,                                                                                   \
      kCannExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Conv<T>);

#define REGISTER_CONV_VERSIONED_TYPED(startver, endver, T) \
  REGISTER_CONV_VERSIONED_TYPED_KERNEL(startver, endver, T)

#define REGISTER_CONV_TYPED(ver, T)  \
  REGISTER_CONV_TYPED_KERNEL(ver, T) \
  REGISTER_CONV_TYPED_COMPUTE(T)

REGISTER_CONV_VERSIONED_TYPED(1, 10, MLFloat16)
REGISTER_CONV_VERSIONED_TYPED(1, 10, float)
REGISTER_CONV_VERSIONED_TYPED(1, 10, double)
REGISTER_CONV_TYPED(11, MLFloat16)
REGISTER_CONV_TYPED(11, float)
REGISTER_CONV_TYPED(11, double)

}  // namespace cann
}  // namespace onnxruntime
