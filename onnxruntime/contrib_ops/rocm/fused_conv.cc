// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <unordered_set>
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

    // Detect if the dimesion changed
    const Tensor* X = context->Input<Tensor>(0);
    const TensorShape& x_shape = X->Shape();
    const auto x_dims = x_shape.GetDims();
    bool input_dims_changed = (Base::s_.last_x_dims.GetDims() != x_dims);

    const Tensor* W = context->Input<Tensor>(1);
    const TensorShape& w_shape = W->Shape();
    auto w_dims = w_shape.AsShapeVector();
    bool w_dims_changed = (Base::s_.last_w_dims.GetDims() != gsl::make_span(w_dims));

    ORT_RETURN_IF_ERROR(Base::UpdateState(context, true));
    if (Base::s_.Y->Shape().Size() == 0) {
      return Status::OK();
    }

    bool has_z = nullptr != Base::s_.z_data;
    bool has_b = nullptr != Base::s_.b_data;

    // TODO: Can be removed once MIOpen supports multiple Bias operators in
    //       single Fusion plan.
    bool should_try_fusion_api = true;
    if (has_z && has_b && this->unsupported_fusion_both_z_b_) {
      should_try_fusion_api = false;
    }

    if (should_try_fusion_api) {
      // Create new Fusion descriptor when dims changed
      if (input_dims_changed || w_dims_changed || !fusion_.plan)  {
        if (Status::OK() != CreateFusionDesc() && has_z && has_b) {
          this->unsupported_fusion_both_z_b_.store(true);
        }
      }
    }

    typedef typename onnxruntime::rocm::ToHipType<T>::MappedType HipT;
    const auto alpha = onnxruntime::rocm::Consts<HipT>::One;
    const auto beta = onnxruntime::rocm::Consts<HipT>::Zero;
    IAllocatorUniquePtr<void> workspace = Base::GetWorkSpace();

    auto fusion_status = CompileOnCurrentHandle();
    if (fusion_.plan && fusion_.fusion_args && miopenStatusSuccess == fusion_status) {
      MIOPEN_RETURN_IF_ERROR(miopenSetOpArgsConvForward(fusion_.fusion_args,
                                                        fusion_.conv_op,
                                                        &alpha,
                                                        &beta,
                                                        Base::s_.w_data));
      if (has_z) {
        MIOPEN_RETURN_IF_ERROR(miopenSetOpArgsBiasForward(fusion_.fusion_args,
                                                          fusion_.bias_z_op,
                                                          &alpha,
                                                          &beta,
                                                          Base::s_.z_data));
      }
      if (has_b) {
        MIOPEN_RETURN_IF_ERROR(miopenSetOpArgsBiasForward(fusion_.fusion_args,
                                                          fusion_.bias_b_op,
                                                          &alpha,
                                                          &beta,
                                                          Base::s_.b_data));
      }
      if (activation_desc_) {
        const float relu_notused = 0.0;
        MIOPEN_RETURN_IF_ERROR(miopenSetOpArgsActivForward(fusion_.fusion_args,
                                                           fusion_.act_op,
                                                           &alpha,
                                                           &beta,
                                                           relu_notused,
                                                           relu_notused,
                                                           relu_notused));
      }
      fusion_status = miopenExecuteFusionPlan(Base::MiopenHandle(),
                                              fusion_.plan,
                                              Base::s_.x_tensor,
                                              Base::s_.x_data,
                                              Base::s_.y_tensor,
                                              Base::s_.y_data,
                                              fusion_.fusion_args);
    }
    // MIOPEN_RETURN_IF_ERROR(fusion_status); // Debug
    if (miopenStatusSuccess != fusion_status) {
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
      MIOPEN_RETURN_IF_ERROR(miopenActivationForward(Base::MiopenHandle(),
                                                     activation_desc_,
                                                     &alpha,
                                                     Base::s_.y_tensor,
                                                     Base::s_.y_data,
                                                     &beta,
                                                     Base::s_.y_tensor,
                                                     Base::s_.y_data));
    }
    if (Base::s_.post_slicing_required) {
      ORT_RETURN_IF_ERROR(onnxruntime::rocm::SliceOutUnwantedOutputSection(
          this->Stream(),
          Base::s_.y_data,
          Base::s_.y_dims_with_adjusted_pads,
          Base::s_.Y->MutableDataRaw(),
          Base::s_.y_dims.GetDims(),
          Base::s_.slice_starts,
          Base::s_.slice_ends,
          Base::s_.slice_axes,
          Base::s_.element_size));
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

  // MIOpen Fusion API
  // TODO: create one fusion descriptor shared by multiple FusedConv
  //       objects
  //
  // Considerations:
  // How to determine two FusedConv objects may share the same fusion
  // descriptor? Hashing x_tensor,conv_desc, etc.?
  struct FusedConvFusionData {
    miopenFusionPlanDescriptor_t plan = nullptr;
    miopenFusionOpDescriptor_t conv_op = nullptr;
    miopenFusionOpDescriptor_t bias_b_op = nullptr;
    miopenFusionOpDescriptor_t bias_z_op = nullptr;
    miopenFusionOpDescriptor_t act_op = nullptr;
    miopenOperatorArgs_t fusion_args = nullptr;

    // TODO: There is a potential problem. miopenHandle_t may be destroyed and
    //       re-created later, sharing the same address. Currently there is any way
    //       to detect it?
    mutable std::unordered_set<miopenHandle_t> compiled_on;

    FusedConvFusionData(const FusedConvFusionData&) = delete;
    FusedConvFusionData& operator= (const FusedConvFusionData&) = delete;

    FusedConvFusionData(FusedConvFusionData&& other) {
      *this = std::move(other);
    }
    FusedConvFusionData& operator=(FusedConvFusionData&& other) {
      std::swap(this->plan, other.plan);
      std::swap(this->fusion_args, other.fusion_args);
      this->conv_op = other.conv_op;
      this->bias_b_op = other.bias_b_op;
      this->bias_z_op = other.bias_z_op;
      this->act_op = other.act_op;
      this->compiled_on = std::move(other.compiled_on);
      return *this;
    }

    FusedConvFusionData() { }
    ~FusedConvFusionData() {
      if (plan) {
        miopenDestroyFusionPlan(plan);
      }
      if (fusion_args) {
          miopenDestroyOperatorArgs(fusion_args);
      }
    }
  };
  mutable FusedConvFusionData fusion_;
  static std::atomic_bool unsupported_fusion_both_z_b_;

  Status CreateFusionDesc() const {
    FusedConvFusionData fusion;
    bool has_z = nullptr != Base::s_.z_data;
    bool has_b = nullptr != Base::s_.b_data;
    MIOPEN_RETURN_IF_ERROR(miopenCreateFusionPlan(&fusion.plan,
                                                  miopenVerticalFusion,
                                                  Base::s_.x_tensor));
    MIOPEN_RETURN_IF_ERROR(miopenCreateOperatorArgs(&fusion.fusion_args));
    MIOPEN_RETURN_IF_ERROR(miopenCreateOpConvForward(fusion.plan,
                                                     &fusion.conv_op,
                                                     Base::s_.conv_desc,
                                                     Base::s_.w_desc));
    if (has_z) {
      MIOPEN_RETURN_IF_ERROR(miopenCreateOpBiasForward(fusion.plan,
                                                       &fusion.bias_z_op,
                                                       Base::s_.z_tensor));
    }
    if (has_b) {
      MIOPEN_RETURN_IF_ERROR(miopenCreateOpBiasForward(fusion.plan,
                                                       &fusion.bias_b_op,
                                                       Base::s_.b_tensor));
    }
    if (activation_desc_) {
      MIOPEN_RETURN_IF_ERROR(miopenCreateOpActivationForward(fusion.plan,
                                                             &fusion.act_op,
                                                             activation_mode_));
    }
    fusion_ = std::move(fusion);
    return Status::OK();
  }

  miopenStatus_t CompileOnCurrentHandle() const {
    if (!fusion_.plan) {
      return miopenStatusNotInitialized;
    }
    auto handle = Base::MiopenHandle();
    auto iter = fusion_.compiled_on.find(handle);
    if (iter != fusion_.compiled_on.end()) {
      return miopenStatusSuccess;
    }
    auto ret = miopenCompileFusionPlan(handle, fusion_.plan);
    if (miopenStatusSuccess == ret) {
      fusion_.compiled_on.insert(handle);
    }
    return miopenStatusSuccess;
  }
};

template <typename T>
std::atomic_bool FusedConv<T>::unsupported_fusion_both_z_b_(false);

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
