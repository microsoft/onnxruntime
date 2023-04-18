// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <unordered_set>
#include <unordered_map>
#include "core/common/status.h"
#include "core/providers/rocm/nn/conv.h"
#include "core/providers/rocm/rocm_common.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

namespace {

// Copied from hipDNN/library/src/hcc_detail/hipdnn_miopen.cpp
miopenStatus_t _miopenAddTensor(
    miopenHandle_t handle,
    const void* alpha,
    const miopenTensorDescriptor_t aDesc,
    const void* A,
    const void* beta,
    const miopenTensorDescriptor_t cDesc,
    void* C,
    const void* zero_scalar) {
  const miopenTensorOp_t tensorOp = miopenTensorOpAdd;
  // Using miopenOpTensor to implement Add operator.
  // opnd2 = Add ( 0.0 * opnd0, alpha * opnd1 ) + beta * opnd2
  return miopenOpTensor(handle, tensorOp,
                        zero_scalar, cDesc, C,
                        alpha, aDesc, A,
                        beta, cDesc, C);
}

}  // namespace

template <uint32_t BASIS = 0x811C9DC5, uint32_t PRIME = 0x01000193>
struct FNVHash {
  uint32_t GetValue() const { return value_; }

  void Hash(const void* in_ptr, size_t nbytes) {
    auto ptr = reinterpret_cast<const uint8_t*>(in_ptr);
    for (size_t i = 0; i < nbytes; ++i) {
      value_ ^= ptr[i];
      value_ *= PRIME;
    }
  }

  template <typename T, typename std::enable_if<std::is_trivially_copyable<T>::value, size_t>::type = 0>
  FNVHash& operator<<(const T& pod) {
    Hash(&pod, sizeof(pod));
    return *this;
  }

  template <typename T>
  FNVHash& operator<<(const std::vector<T>& pod_array) {
    for (const auto& pod : pod_array) {
      (*this) << pod;
    }
    return *this;
  }

  void HashTensor(miopenTensorDescriptor_t tdesc) {
    int size = 0;
    miopenGetTensorDescriptorSize(tdesc, &size);
    (*this) << size;
    std::vector<int> dims(size);
    std::vector<int> strides(size);
    miopenDataType_t dtype;
    miopenGetTensorDescriptor(tdesc, &dtype, dims.data(), strides.data());
    (*this) << dtype;
    (*this) << dims;
    (*this) << strides;
  }

  void HashConvolutionDescriptor(miopenConvolutionDescriptor_t cdesc) {
    int spatial_dim = 1;
#if ROCM_VERSION >= 50500
    miopenGetConvolutionSpatialDim(cdesc, &spatial_dim);
#else
    // Previous versions of MIOpen doesn't provide API to probe the dimension of a
    // miopenConvolutionDescriptor_t, so we have to guess.
    // This algorithm is based on a specific behavior of miopenGetConvolutionNdDescriptor,
    //  which fails when requestedSpatialDim > the convolution's spatial dimension
    constexpr const int kMaxSpatialDim = 5;
    std::vector<int> pads{kMaxSpatialDim};
    std::vector<int> strides{kMaxSpatialDim};
    std::vector<int> dilations{kMaxSpatialDim};
    miopenConvolutionMode_t mode;
    bool spatial_dim_guessed = false;
    for (int i = 0; i < kMaxSpatialDim; i++) {
      if (miopenStatusSuccess == miopenGetConvolutionNdDescriptor(
                                     cdesc, i, &spatial_dim, pads.data(), strides.data(), dilations.data(), &mode)) {
        spatial_dim_guessed = true;
        break;
      }
    }
    ORT_ENFORCE(spatial_dim_guessed, "Failed to guess the actual spatial dimension");
    // Remove the extra dimension
    pads.resize(spatial_dim);
    strides.resize(spatial_dim);
    dilations.resize(spatial_dim);
    (*this) << spatial_dim;
    (*this) << pads;
    (*this) << strides;
    (*this) << dilations;
#endif
  }

 private:
  uint32_t value_ = BASIS;
};

template <typename T>
class FusedConv : public onnxruntime::rocm::Conv<T, false> {
 public:
  using Base = onnxruntime::rocm::Conv<T, false>;
  FusedConv(const OpKernelInfo& info) : onnxruntime::rocm::Conv<T, false>(info) {
    std::string activation;
    ORT_THROW_IF_ERROR(info.GetAttr<std::string>("activation", &activation));
    ORT_THROW_IF_ERROR(MapMode(activation));
    MIOPEN_CALL_THROW(miopenCreateActivationDescriptor(&activation_desc_));
    MIOPEN_CALL_THROW(miopenSetActivationDescriptor(activation_desc_, activation_mode_, 0.0, 0.0, 0.0));
  }

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(FusedConv);

  ~FusedConv() {
    if (activation_desc_) {
      MIOPEN_CALL_THROW(miopenDestroyActivationDescriptor(activation_desc_));
      activation_desc_ = nullptr;
    }
  }

  Status ComputeInternal(OpKernelContext* context) const override {
    std::lock_guard<OrtMutex> lock(Base::s_.mutex);

    ORT_RETURN_IF_ERROR(Base::UpdateState(context, true));
    if (Base::s_.Y->Shape().Size() == 0) {
      return Status::OK();
    }

    bool has_z = nullptr != Base::s_.z_data;
    bool has_b = nullptr != Base::s_.b_data;
    auto factory = [this](FusedConvFusionData& fusion) {
      return this->DoCreateFusionDesc(this->Node().Name(), fusion);
    };
    auto& cached_item = plan_cache_.FindOrCreateFusionPlanCache(Hash(),
                                                                factory);
    bool should_try_fusion_api = cached_item.Validate(this->GetMiopenHandle(context));

    typedef typename onnxruntime::rocm::ToHipType<T>::MappedType HipT;
    const auto alpha = onnxruntime::rocm::Consts<HipT>::One;
    const auto beta = onnxruntime::rocm::Consts<HipT>::Zero;
    IAllocatorUniquePtr<void> workspace = Base::GetWorkSpace(context->GetComputeStream());
    miopenStatus_t fusion_status = miopenStatusNotInitialized;

    if (should_try_fusion_api) {
      auto& fusion_info = *cached_item.fusion;
      MIOPEN_RETURN_IF_ERROR(miopenSetOpArgsConvForward(fusion_info.fusion_args,
                                                        fusion_info.conv_op,
                                                        &alpha,
                                                        &beta,
                                                        Base::s_.w_data));
      if (has_z) {
        MIOPEN_RETURN_IF_ERROR(miopenSetOpArgsBiasForward(fusion_info.fusion_args,
                                                          fusion_info.bias_z_op,
                                                          &alpha,
                                                          &beta,
                                                          Base::s_.z_data));
      }
      if (has_b) {
        MIOPEN_RETURN_IF_ERROR(miopenSetOpArgsBiasForward(fusion_info.fusion_args,
                                                          fusion_info.bias_b_op,
                                                          &alpha,
                                                          &beta,
                                                          Base::s_.b_data));
      }
      if (activation_desc_) {
        const float relu_notused = 0.0;
        MIOPEN_RETURN_IF_ERROR(miopenSetOpArgsActivForward(fusion_info.fusion_args,
                                                           fusion_info.act_op,
                                                           &alpha,
                                                           &beta,
                                                           relu_notused,
                                                           relu_notused,
                                                           relu_notused));
      }
      fusion_status = miopenExecuteFusionPlan(this->GetMiopenHandle(context),
                                              fusion_info.plan,
                                              Base::s_.x_tensor,
                                              Base::s_.x_data,
                                              Base::s_.y_tensor,
                                              Base::s_.y_data,
                                              fusion_info.fusion_args);
    }
    if (miopenStatusSuccess != fusion_status) {
      MIOPEN_RETURN_IF_ERROR(miopenConvolutionForward(this->GetMiopenHandle(context),
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
        MIOPEN_RETURN_IF_ERROR(_miopenAddTensor(this->GetMiopenHandle(context),
                                                &alpha, Base::s_.b_tensor, Base::s_.b_data,
                                                &alpha, Base::s_.y_tensor, Base::s_.y_data,
                                                &beta));
      }
      if (has_z) {
        MIOPEN_RETURN_IF_ERROR(_miopenAddTensor(this->GetMiopenHandle(context),
                                                &alpha, Base::s_.z_tensor, Base::s_.z_data,
                                                &alpha, Base::s_.y_tensor, Base::s_.y_data,
                                                &beta));
      }
      MIOPEN_RETURN_IF_ERROR(miopenActivationForward(this->GetMiopenHandle(context),
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
          this->Stream(context),
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
      return ORT_MAKE_STATUS(
          StatusCategory::ONNXRUNTIME, StatusCode::INVALID_ARGUMENT,
          "unsupported conv activation mode \"", activaton_mode, "\"");
    }
    return Status::OK();
  }
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
    FusedConvFusionData& operator=(const FusedConvFusionData&) = delete;

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

    FusedConvFusionData() {}
    ~FusedConvFusionData() {
      if (plan) {
        miopenDestroyFusionPlan(plan);
      }
      if (fusion_args) {
        miopenDestroyOperatorArgs(fusion_args);
      }
    }
  };

  struct FusionPlanCacheItem {
    std::unique_ptr<FusedConvFusionData> fusion;
    Status creation_result;
    // TODO: Add a timestamp for eviction
    // std::chrono::time_point<std::chrono::high_resolution_clock> last_access;

    FusionPlanCacheItem() {
    }

    miopenStatus_t CompileOnHandle(miopenHandle_t handle) const {
      if (!fusion->plan) {
        return miopenStatusNotInitialized;
      }
      auto iter = fusion->compiled_on.find(handle);
      if (iter != fusion->compiled_on.end()) {
        return miopenStatusSuccess;
      }
      auto ret = miopenCompileFusionPlan(handle, fusion->plan);
      if (miopenStatusSuccess == ret) {
        fusion->compiled_on.insert(handle);
      }
      return miopenStatusSuccess;
    }

    bool Validate(miopenHandle_t handle) const {
      if (Status::OK() != creation_result) {
        return false;
      }
      if (!fusion || !fusion->plan || !fusion->fusion_args) {
        return false;
      }
      auto compiling_status = CompileOnHandle(handle);
      if (miopenStatusSuccess != compiling_status) {
        return false;
      }

      return true;
    }
  };

  struct FusionPlanCache {
    mutable OrtMutex mutex;
    using HashKey = uint32_t;
    std::unordered_map<HashKey, FusionPlanCacheItem> cache_directory_;

    FusionPlanCache() {
    }

    FusionPlanCacheItem& FindOrCreateFusionPlanCache(HashKey key,
                                                     std::function<Status(FusedConvFusionData& fusion)> factory) {
      std::lock_guard<OrtMutex> lock(mutex);
      auto iter = cache_directory_.find(key);
      if (iter == cache_directory_.end()) {
        cache_directory_[key].fusion = std::make_unique<FusedConvFusionData>();
        cache_directory_[key].creation_result = factory(*cache_directory_[key].fusion);
        if (Status::OK() != cache_directory_[key].creation_result) {
          cache_directory_[key].fusion.reset();
        }
      }
      return cache_directory_[key];
    }
  };

  static FusionPlanCache plan_cache_;

  Status DoCreateFusionDesc(const std::string& node_name, FusedConvFusionData& fusion) const {
    bool has_z = nullptr != Base::s_.z_data;
    bool has_b = nullptr != Base::s_.b_data;
    MIOPEN_RETURN_IF_ERROR(miopenCreateFusionPlan(&fusion.plan,
                                                  miopenVerticalFusion,
                                                  Base::s_.x_tensor));
    MIOPEN_RETURN_IF_ERROR(miopenCreateOperatorArgs(&fusion.fusion_args));
    auto status = miopenCreateOpConvForward(fusion.plan, &fusion.conv_op, Base::s_.conv_desc, Base::s_.w_desc);
    if (status == miopenStatusUnsupportedOp) {
      auto msg = MakeString("MIOpen does not support the conv fusion for node \"",
                            node_name, "\", fallback to unfused implementation.");
      LOGS_DEFAULT(WARNING) << msg;
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, msg);
    }
    MIOPEN_RETURN_IF_ERROR(status);

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
    return Status::OK();
  }

  uint32_t Hash() const {
    FNVHash hash;
    bool has_z = nullptr != Base::s_.z_data;
    bool has_b = nullptr != Base::s_.b_data;
    hash.HashTensor(Base::s_.x_tensor);
    hash.HashConvolutionDescriptor(Base::s_.conv_desc);
    hash.HashTensor(Base::s_.w_desc);
    if (has_z) {
      hash.HashTensor(Base::s_.z_tensor);
    }
    if (has_b) {
      hash.HashTensor(Base::s_.b_tensor);
    }
    if (activation_desc_) {
      hash << static_cast<int32_t>(activation_mode_);
    }
    return hash.GetValue();
  }
};

template <typename T>
typename FusedConv<T>::FusionPlanCache FusedConv<T>::plan_cache_;

#define REGISTER_KERNEL_TYPED(T)                                                           \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      FusedConv,                                                                           \
      kMSDomain,                                                                           \
      1,                                                                                   \
      T,                                                                                   \
      kRocmExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      FusedConv<T>);

REGISTER_KERNEL_TYPED(float);
REGISTER_KERNEL_TYPED(MLFloat16);
}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
