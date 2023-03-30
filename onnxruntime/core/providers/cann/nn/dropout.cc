// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cann/nn/dropout.h"

namespace onnxruntime {
namespace cann {

namespace {

constexpr float default_ratio{0.5f};

template <typename T2>
float GetRatioOrDefault(const Tensor* ratio) {
  if (ratio) {
    ORT_ENFORCE(ratio->Shape().Size() == 1, "ratio input should have a single value.");
    const float ratio_value = *ratio->Data<T2>();
    ORT_ENFORCE(0.0f <= ratio_value && ratio_value < 1.0f, "ratio must be in the range [0, 1)");
    return ratio_value;
  }

  return default_ratio;
}

}  // namespace

template <typename T1, typename T2>
Status Dropout<T1, T2>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& X_shape = X->Shape();

  const Tensor* ratio = context->Input<Tensor>(1);
  const float ratio_value = GetRatioOrDefault<T2>(ratio);

  const Tensor* training_mode = context->Input<Tensor>(2);

  auto Y = context->Output(0, X_shape);
  auto mask = context->Output(1, X_shape);

  if (ratio_value == 0.f || !training_mode || !(*(training_mode->Data<bool>()))) {
    const void* X_data = X->DataRaw();
    void* Y_data = Y->MutableDataRaw();

    if (Y_data != X_data) {
      CANN_RETURN_IF_ERROR(aclrtMemcpyAsync(Y_data, Y->SizeInBytes(), X_data, Y->SizeInBytes(),
                                            ACL_MEMCPY_DEVICE_TO_DEVICE, Stream()));
    }

    if (mask) {
      CANN_RETURN_IF_ERROR(aclrtMemsetAsync(mask->MutableData<bool>(), mask->SizeInBytes(), true,
                                            mask->SizeInBytes(), Stream()));
    }
  } else {
    IAllocatorUniquePtr<void> pmask{};
    IAllocatorUniquePtr<void> pseed = GetScratchBuffer<void>(sizeof(float));

    void* mask_data = nullptr;
    if (mask) {
      mask_data = mask->MutableDataRaw();
    } else {
      pmask = GetScratchBuffer<void>(X_shape.Size() * sizeof(bool));
      mask_data = pmask.get();
    }

    RandomGenerator& generator = generator_ != nullptr ? *generator_.get() : RandomGenerator::Default();
    float seed = static_cast<float>(generator.NextSeed());
    // TODO(FFFrog): use aclrtMemcpyAsyn to improve performance later.
    CANN_RETURN_IF_ERROR(aclrtMemcpy(pseed.get(), sizeof(float), &seed, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));

    TensorShape shape{1};

    const aclDataType aclType = getACLType<T1>();
    aclFormat format = ACL_FORMAT_ND;

    CannPreparation prepare;

    CANN_RETURN_IF_ERROR(aclopSetAttrInt(prepare.opAttr_, "p", static_cast<float>(ratio_value)));

    ORT_TRY {
      CANN_PREPARE_INPUTDESC(prepare, aclType, X_shape.NumDimensions(), X_shape.GetDims().data(), format);
      CANN_PREPARE_INPUTDESC(prepare, ACL_FLOAT, shape.NumDimensions(), shape.GetDims().data(), format);

      CANN_PREPARE_OUTPUTDESC(prepare, aclType, X_shape.NumDimensions(), X_shape.GetDims().data(), format);
      CANN_PREPARE_OUTPUTDESC(prepare, ACL_BOOL, X_shape.NumDimensions(), X_shape.GetDims().data(), format);
      CANN_PREPARE_OUTPUTDESC(prepare, ACL_FLOAT, shape.NumDimensions(), shape.GetDims().data(), format);

      CANN_PREPARE_INPUTBUFFER(prepare, const_cast<void*>(X->DataRaw()), X->SizeInBytes());
      CANN_PREPARE_INPUTBUFFER(prepare, pseed.get(), sizeof(float));

      CANN_PREPARE_OUTPUTBUFFER(prepare, Y->MutableDataRaw(), Y->SizeInBytes());
      CANN_PREPARE_OUTPUTBUFFER(prepare, mask_data, X_shape.Size() * sizeof(float));
      CANN_PREPARE_OUTPUTBUFFER(prepare, pseed.get(), sizeof(float));
    }
    ORT_CATCH(const std::exception& e) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
    }

    CANN_RETURN_IF_ERROR(aclopCompileAndExecute("DropoutV2",
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
                                                Stream()));
  }

  return Status::OK();
}

#define REGISTER_DROPOUT_VERSIONED_TYPED_KERNEL(startver, endver, T1, T2) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                \
      Dropout,                                                            \
      kOnnxDomain,                                                        \
      startver,                                                           \
      endver,                                                             \
      T1##_##T2,                                                          \
      kCannExecutionProvider,                                             \
      (*KernelDefBuilder::Create())                                       \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T1>())         \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T2>())        \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>())      \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                         \
          .InputMemoryType(OrtMemTypeCPUInput, 2),                        \
      Dropout<T1, T2>);

#define REGISTER_DROPOUT_TYPED_KERNEL(ver, T1, T2)                   \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                     \
      Dropout,                                                       \
      kOnnxDomain,                                                   \
      ver,                                                           \
      T1##_##T2,                                                     \
      kCannExecutionProvider,                                        \
      (*KernelDefBuilder::Create())                                  \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T1>())    \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T2>())   \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>()) \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                    \
          .InputMemoryType(OrtMemTypeCPUInput, 2),                   \
      Dropout<T1, T2>);

REGISTER_DROPOUT_VERSIONED_TYPED_KERNEL(12, 12, MLFloat16, MLFloat16)
REGISTER_DROPOUT_VERSIONED_TYPED_KERNEL(12, 12, float, float)

REGISTER_DROPOUT_TYPED_KERNEL(13, MLFloat16, MLFloat16)
REGISTER_DROPOUT_TYPED_KERNEL(13, float, float)

}  // namespace cann
}  // namespace onnxruntime
