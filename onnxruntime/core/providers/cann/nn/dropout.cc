// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cann/nn/dropout.h"

namespace onnxruntime {
namespace cann {

namespace {

template <typename T>
struct GetRatioDataImpl {
  void operator()(const Tensor* ratio, float& ratio_data) const {
    ratio_data = static_cast<float>(*(ratio->Data<T>()));
    ORT_ENFORCE(ratio_data >= 0.0f && ratio_data < 1.0f, "ratio_data is outside range [0, 1)");
  }
};

template <typename T>
struct DropoutComputeImpl {
  void operator()(const Tensor* X, Tensor* Y, float ratio_data, void* mask_data, aclrtStream stream) const {
    const aclDataType aclType = getACLType<T>();
    aclFormat format = ACL_FORMAT_ND;

    TensorShape shape{1};
    bool train_mode = true;

    CannPreparation prepare;

    CANN_PREPARE_INPUTDESC(prepare, aclType, X->Shape().NumDimensions(), X->Shape().GetDims().data(), format);
    CANN_PREPARE_INPUTDESC(prepare, ACL_FLOAT, shape.NumDimensions(), shape.GetDims().data(), format);
    CANN_PREPARE_INPUTDESC(prepare, ACL_BOOL, shape.NumDimensions(), shape.GetDims().data(), format);
    CANN_PREPARE_OUTPUTDESC(prepare, aclType, Y->Shape().NumDimensions(), Y->Shape().GetDims().data(), format);
    CANN_PREPARE_OUTPUTDESC(prepare, ACL_BOOL, X->Shape().NumDimensions(), X->Shape().GetDims().data(), format);

    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<T*>(X->template Data<T>()), X->SizeInBytes());
    CANN_PREPARE_INPUTBUFFER(prepare, &ratio_data, sizeof(float));
    CANN_PREPARE_INPUTBUFFER(prepare, &train_mode, sizeof(bool));
    CANN_PREPARE_OUTPUTBUFFER(prepare, Y->template MutableData<T>(), Y->SizeInBytes());
    CANN_PREPARE_OUTPUTBUFFER(prepare, mask_data, X->Shape().Size());

    CANN_CALL_THROW(aclopCompileAndExecute("Dropout",
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
                                           stream));
  }
};

}  // namespace

Status Dropout::ComputeInternal(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  if (!X) return Status(common::ONNXRUNTIME, common::FAIL, "X Input is not available.");
  const TensorShape& shape = X->Shape();
  const int64_t N = shape.Size();

  auto Y = context->Output(0, shape);

  Tensor* mask = context->Output(1, shape);
  ORT_ENFORCE(!mask || mask->Shape().Size() == N);

  float ratio_data = default_ratio_;
  auto ratio = context->Input<Tensor>(1);
  if (ratio) {
    utils::MLTypeCallDispatcher<float, MLFloat16, double, BFloat16> t_disp(ratio->GetElementType());
    t_disp.Invoke<GetRatioDataImpl>(ratio, ratio_data);
  }

  const Tensor* training_mode = context->Input<Tensor>(2);
  if (ratio_data == 0.f || !training_mode || !(*(training_mode->Data<bool>()))) {
    const void* X_data = X->DataRaw();
    void* Y_data = Y->MutableDataRaw();

    if (Y_data != X_data) {
      CANN_RETURN_IF_ERROR(aclrtMemcpyAsync(Y_data, Y->SizeInBytes(), X_data, X->SizeInBytes(),
                                            ACL_MEMCPY_DEVICE_TO_DEVICE, Stream()));
    }

    if (mask) {
      CANN_RETURN_IF_ERROR(aclrtMemsetAsync(mask->MutableData<bool>(), mask->SizeInBytes(), true,
                                            N * sizeof(bool), Stream()));
    }

    return Status::OK();
  }

  IAllocatorUniquePtr<void> temp_mask_buffer{};  // buffer to use if mask is not provided
  void* const mask_data = [this, N, mask, &temp_mask_buffer]() {
    if (mask) return mask->MutableDataRaw();
    temp_mask_buffer =
        GetScratchBuffer<void>(N * sizeof(bool));
    return temp_mask_buffer.get();
  }();

  ORT_TRY {
    utils::MLTypeCallDispatcher<float, MLFloat16, double, BFloat16> t_disp(X->GetElementType());
    t_disp.Invoke<DropoutComputeImpl>(X, Y, ratio_data, mask_data, Stream());
  }
  ORT_CATCH(const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
  }

  return Status::OK();
}

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Dropout, kOnnxDomain, 12, 12, kCannExecutionProvider,
                                  (*KernelDefBuilder::Create())
                                      .TypeConstraint("T", DataTypeImpl::AllIEEEFloatTensorTypes())
                                      .TypeConstraint("T1", DataTypeImpl::AllIEEEFloatTensorTypes())
                                      .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>())
                                      .InputMemoryType(OrtMemTypeCPUInput, 1)
                                      .InputMemoryType(OrtMemTypeCPUInput, 2),
                                  Dropout);

ONNX_OPERATOR_KERNEL_EX(Dropout, kOnnxDomain, 13, kCannExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .TypeConstraint("T", BuildKernelDefConstraints<MLFloat16, float, double, BFloat16>())
                            .TypeConstraint("T1", BuildKernelDefConstraints<MLFloat16, float, double>())
                            .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>())
                            .InputMemoryType(OrtMemTypeCPUInput, 1)
                            .InputMemoryType(OrtMemTypeCPUInput, 2),
                        Dropout);

}  // namespace cann
}  // namespace onnxruntime
