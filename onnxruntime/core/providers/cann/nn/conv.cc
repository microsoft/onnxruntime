// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <unordered_map>
#include "core/providers/cann/nn/conv.h"

using onnxruntime::common::Status;
namespace onnxruntime {
namespace cann {

using ConvPadVector = ConvAttributes::ConvPadVector;

template <typename T>
Status Conv<T>::ComputeInternal(OpKernelContext* ctx) const {
  const auto* X = ctx->Input<Tensor>(0);
  const auto* W = ctx->Input<Tensor>(1);
  const auto* B = ctx->Input<Tensor>(2);
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

  if (dilations.size() < 4) {
    dilations.insert(dilations.begin(), {1, 1});
  }
  if (strides.size() < 4) {
    strides.insert(strides.begin(), {1, 1});
  }

  std::unordered_map<AutoPadType, const char*> padding_mode = {
      {AutoPadType::NOTSET, "NOTSET"},
      {AutoPadType::SAME_UPPER, "SAME_UPPER"},
      {AutoPadType::SAME_LOWER, "SAME_LOWER"},
      {AutoPadType::VALID, "VALID"}};

  std::string opname = X->Shape().NumDimensions() > 4 ? "Conv3D" : "Conv2D";
  bool is_trans_2d = X->Shape().NumDimensions() > 4 ? false : true;

  const aclDataType aclType = getACLType<T>();
  aclFormat format = ACL_FORMAT_NCHW;

  CannPreparation prepare;

  CANN_RETURN_IF_ERROR(aclopSetAttrListInt(prepare.opAttr_, "strides", strides.size(), strides.data()));
  CANN_RETURN_IF_ERROR(aclopSetAttrListInt(prepare.opAttr_, "pads", pads.size(), pads.data()));
  CANN_RETURN_IF_ERROR(aclopSetAttrListInt(prepare.opAttr_, "dilations", dilations.size(), dilations.data()));
  CANN_RETURN_IF_ERROR(aclopSetAttrInt(prepare.opAttr_, "group", conv_attrs_.group));
  CANN_RETURN_IF_ERROR(aclopSetAttrString(prepare.opAttr_, "auto_pad", padding_mode[conv_attrs_.auto_pad]));
  CANN_RETURN_IF_ERROR(aclopSetAttrInt(prepare.opAttr_, "dim_size", X->Shape().NumDimensions()));
  CANN_RETURN_IF_ERROR(aclopSetAttrBool(prepare.opAttr_, "trans_2d", is_trans_2d));

  ORT_TRY {
    CANN_PREPARE_INPUTDESC(prepare, aclType, X->Shape().NumDimensions(), X->Shape().GetDims().data(), format);
    CANN_PREPARE_INPUTDESC(prepare, aclType, W->Shape().NumDimensions(), W->Shape().GetDims().data(), format);
    if (ctx->InputCount() >= 3)
      CANN_PREPARE_INPUTDESC(prepare, aclType, B->Shape().NumDimensions(), B->Shape().GetDims().data(), ACL_FORMAT_ND);
    else
      CANN_PREPARE_INPUTDESC(prepare, ACL_DT_UNDEFINED, 0, nullptr, ACL_FORMAT_UNDEFINED);

    CANN_PREPARE_OUTPUTDESC(prepare, aclType, Y->Shape().NumDimensions(), Y->Shape().GetDims().data(), format);

    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<void*>(X->DataRaw()), X->SizeInBytes());
    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<void*>(W->DataRaw()), W->SizeInBytes());
    if (ctx->InputCount() >= 3)
      CANN_PREPARE_INPUTBUFFER(prepare, const_cast<void*>(B->DataRaw()), B->SizeInBytes());
    else
      CANN_PREPARE_INPUTBUFFER(prepare, nullptr, 0);

    CANN_PREPARE_OUTPUTBUFFER(prepare, Y->MutableData<T>(), Y->SizeInBytes());
  }
  ORT_CATCH(const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
  }

  CANN_RETURN_IF_ERROR(aclopCompileAndExecute(opname.c_str(),
                                              prepare.inputDesc_.size(),
                                              prepare.inputDesc_.data(),
                                              prepare.inputBuffers_.data(),
                                              prepare.outputDesc_.size(),
                                              prepare.outputDesc_.data(),
                                              prepare.outputBuffers_.data(),
                                              prepare.opAttr_,
                                              ACL_ENGINE_SYS,
                                              ACL_COMPILE_SYS,
                                              NULL,
                                              Stream(ctx)));

  return Status::OK();
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

REGISTER_CONV_VERSIONED_TYPED_KERNEL(1, 10, MLFloat16)
REGISTER_CONV_VERSIONED_TYPED_KERNEL(1, 10, float)
REGISTER_CONV_TYPED_KERNEL(11, MLFloat16)
REGISTER_CONV_TYPED_KERNEL(11, float)

}  // namespace cann
}  // namespace onnxruntime
