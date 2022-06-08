// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/rocm/nn/conv.h"
#include "core/providers/rocm/rocm_common.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

namespace {

// Copied from hipDNN/library/src/hcc_detail/hipdnn_miopen.cpp
miopenStatus_t _miopenAddTensor(
    miopenHandle_t handle,
    const void *alpha,
    const miopenTensorDescriptor_t aDesc,
    const void *A,
    const void *beta,
    const miopenTensorDescriptor_t cDesc,
    void *C,
    const void* zero_scalar)
{
    const miopenTensorOp_t tensorOp = miopenTensorOpAdd;
    // opnd2 = Add ( 0.0 * opnd0, alpha * opnd1 ) + alpha * opnd2
    return miopenOpTensor(handle, tensorOp,
                          zero_scalar, cDesc, C,
                          alpha, aDesc, A,
                          alpha, cDesc, C);
}

}

template <typename T>
class FusedConv : public onnxruntime::rocm::Conv<T> {
 public:
  using Base = onnxruntime::rocm::Conv<T>;
  FusedConv(const OpKernelInfo& info) : onnxruntime::rocm::Conv<T>(info) {
    std::string activation;
    if (info.GetAttr<std::string>("activation", &activation) == Status::OK() &&
        MapMode(activation) == Status::OK() &&
        miopenCreateActivationDescriptor(&activation_desc_) == miopenStatusSuccess) {

      status_ = miopenSetActivationDescriptor(activation_desc_,
                                              activation_mode_,
                                              0.0, 0.0, 0.0);
    }
  }

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(FusedConv);

  ~FusedConv() {
    if (activation_desc_) {
      miopenDestroyActivationDescriptor(activation_desc_);
      status_ = miopenStatusNotInitialized;
      activation_desc_ = nullptr;
    }
  }

  Status ComputeInternal(OpKernelContext* context) const override {
    MIOPEN_RETURN_IF_ERROR(status_);
    std::lock_guard<OrtMutex> lock(Base::s_.mutex);
    ORT_RETURN_IF_ERROR(Base::UpdateState(context, true));
    if (Base::s_.Y->Shape().Size() == 0) {
      return Status::OK();
    }
    bool has_z = nullptr != Base::s_.z_data;
    bool has_b = nullptr != Base::s_.b_data;
    typedef typename onnxruntime::rocm::ToHipType<T>::MappedType HipT;
    const auto alpha = onnxruntime::rocm::Consts<HipT>::One;
    const auto beta = onnxruntime::rocm::Consts<HipT>::Zero;
    IAllocatorUniquePtr<void> workspace = Base::GetWorkSpace();

    MIOPEN_RETURN_IF_ERROR(miopenConvolutionForward(Base::MiopenHandle(),
                           &alpha,
                           Base::s_.x_tensor,
                           Base::s_.x_data,
                           Base::s_.w_desc,
                           Base::s_.w_data,
                           Base::s_.conv_desc,
                           Base::s_.fwd_algo,
                           &beta,
                           Base::s_.y_tensor,
                           Base::s_.y_data,
                           workspace.get(),
                           Base::s_.workspace_bytes));
    if (has_b) {
        MIOPEN_RETURN_IF_ERROR(_miopenAddTensor(Base::MiopenHandle(),
                                                &alpha, Base::s_.b_tensor, Base::s_.b_data,
                                                &alpha, Base::s_.y_tensor, Base::s_.y_data,
                                                &beta));
    }
    if (has_z) {
        MIOPEN_RETURN_IF_ERROR(_miopenAddTensor(Base::MiopenHandle(),
                                                &alpha, Base::s_.z_tensor, Base::s_.z_data,
                                                &alpha, Base::s_.y_tensor, Base::s_.y_data,
                                                &beta));
    }
    MIOPEN_RETURN_IF_ERROR(miopenActivationForward(Base::MiopenHandle(), activation_desc_, &alpha, Base::s_.y_tensor,
                                                   Base::s_.y_data, &beta, Base::s_.y_tensor, Base::s_.y_data));
    if (Base::s_.post_slicing_required) {
      ORT_RETURN_IF_ERROR(onnxruntime::rocm::SliceOutUnwantedOutputSection(
          this->Stream(), Base::s_.y_data, Base::s_.y_dims_with_adjusted_pads, Base::s_.Y->MutableDataRaw(),
          Base::s_.y_dims.GetDims(), Base::s_.slice_starts, Base::s_.slice_ends, Base::s_.slice_axes, Base::s_.element_size));
    }
    return Status::OK();
  }

 private:
  Status MapMode(const std::string& activaton_mode) {
    if (activaton_mode == "Relu") {
      activation_mode_ = miopenActivationMode_t::miopenActivationRELU;
    } else {
      return Status(common::StatusCategory::ONNXRUNTIME,
                    common::StatusCode::INVALID_ARGUMENT,
                    "unsupported conv activation mode");
    }
    return Status::OK();
  }
  miopenStatus_t status_ = miopenStatusNotInitialized;
  miopenActivationMode_t activation_mode_;
  miopenActivationDescriptor_t activation_desc_ = nullptr;
};

ONNX_OPERATOR_TYPED_KERNEL_EX(
    FusedConv,
    kMSDomain,
    1,
    float,
    kRocmExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    FusedConv<float>);

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
