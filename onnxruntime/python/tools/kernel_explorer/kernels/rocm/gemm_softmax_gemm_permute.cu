// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "pybind11/stl.h"

#include "contrib_ops/rocm/bert/batched_gemm_softmax_gemm_permute_pipelines.cuh"
#include "core/providers/rocm/tunable/rocm_tunable.h"
#include "python/tools/kernel_explorer/device_array.h"
#include "python/tools/kernel_explorer/kernel_explorer_interface.h"

#include <vector>

namespace py = pybind11;

using namespace onnxruntime::contrib::rocm;

namespace onnxruntime {

template <typename T>
class IGemmSoftmaxGemmPermuteKernelExplorer : public IKernelExplorer {
 public:
  IGemmSoftmaxGemmPermuteKernelExplorer(
      int64_t batch,
      int64_t seqlen,
      int64_t total_seqlen,
      std::optional<int64_t> max_seqlen,
      int64_t num_heads,
      int64_t head_size,
      int64_t mask_dim,
      double scale,
      contrib::AttentionQkvFormat qkv_format,
      DeviceArray& Q,
      std::optional<DeviceArray>& K,
      std::optional<DeviceArray>& V,
      std::optional<DeviceArray>& attn_bias,
      std::optional<DeviceArray>& attn_mask,
      DeviceArray& out) {
    ROCBLAS_CALL_THROW(rocblas_create_handle(&rocblas_handle_));

    attn_.batch_size = batch;
    attn_.sequence_length = seqlen;
    // NOTE: This test wrapper does not support past present concat, then past_sequence_length = 0 always holds.
    // Thus, total_sequence_length = past_sequence_length + kv_sequence_length further implies
    // total_sequence_length == kv_sequence_length
    attn_.kv_sequence_length = total_seqlen;
    attn_.past_sequence_length = 0;
    attn_.total_sequence_length = total_seqlen;
    attn_.max_sequence_length = 0;
    attn_.hidden_size = num_heads * head_size;
    attn_.head_size = head_size;
    attn_.v_hidden_size = attn_.hidden_size;  // Q,K,V hidden size must agree now
    attn_.v_head_size = attn_.head_size;      // Q,K,V hidden size must agree now
    attn_.num_heads = num_heads;
    attn_.is_unidirectional = false;
    attn_.past_present_share_buffer = false;
    attn_.do_rotary = false;
    attn_.mask_filter_value = -10000.0f;
    attn_.scale = scale;
    if (mask_dim == 0) {
      attn_.mask_type = contrib::MASK_NONE;
    } else if (mask_dim == 2) {
      attn_.mask_type = contrib::MASK_2D_KEY_PADDING;
    } else if (mask_dim == 3) {
      attn_.mask_type = contrib::MASK_3D_ATTENTION;
    } else if (mask_dim == 4) {
      attn_.mask_type = contrib::MASK_4D_MEGATRON;
    } else {
      ORT_ENFORCE(false, "mask type not supported");
    }
    attn_.qkv_format = qkv_format;
    switch (qkv_format) {
      case contrib::Q_K_V_BNSH:
      case contrib::Q_K_V_BSNH:
        attn_.mode = contrib::rocm::QFMT_KFMT_VFMT_NONE_NONE_NONE_NONE;
        break;
      case contrib::Q_KV_BSNH_BSN2H:
        attn_.mode = contrib::rocm::BSNH_BLN2H_NONE_NONE_NONE_NONE_NONE;
        break;
      case contrib::QKV_BSN3H:
        attn_.mode = contrib::rocm::BLN3H_NONE_NONE_NONE_NONE_NONE_NONE;
        break;
      default:
        ORT_NOT_IMPLEMENTED("qkv_format ", qkv_format, " is not implemented");
    }

    device_prop = GetEp()->GetDeviceProp();

    params_.tuning_ctx = TuningContext();
    params_.stream = Stream();
    params_.handle = rocblas_handle_;
    params_.attention = &attn_;
    params_.device_prop = &device_prop;
    params_.scale = scale;

    std::tie(params_.q_buffer, params_.k_buffer, params_.v_buffer) = ConvertToOffsetedBufferViews<T>(
        &attn_, Q.ptr(), K.has_value() ? K->ptr() : nullptr, V.has_value() ? V->ptr() : nullptr);

    if (attn_bias.has_value()) {
      params_.bias_buffer = reinterpret_cast<T*>(attn_bias->ptr());
    }
    if (attn_mask.has_value()) {
      params_.mask_index_buffer = reinterpret_cast<int*>(attn_mask->ptr());
      if (mask_dim == 2) {
        params_.mask_index_dims = {batch, total_seqlen};
      } else if (mask_dim == 3) {
        params_.mask_index_dims = {batch, seqlen, total_seqlen};
      } else if (mask_dim == 4) {
        ORT_ENFORCE(max_seqlen.has_value());
        attn_.max_sequence_length = max_seqlen.value();
        ORT_ENFORCE(attn_.max_sequence_length >= seqlen);
        attn_.past_sequence_length = attn_.max_sequence_length - seqlen;
        params_.mask_index_dims = {batch, 1, attn_.max_sequence_length, attn_.max_sequence_length};
      }
    }
    params_.out_buffer = reinterpret_cast<T*>(out.ptr());
  }

  ~IGemmSoftmaxGemmPermuteKernelExplorer() {
    ROCBLAS_CALL_THROW(rocblas_destroy_handle(rocblas_handle_));
  }

  void SetWorkspace(size_t num_bytes) {
    void* ptr;
    HIP_CALL_THROW(hipMalloc(&ptr, num_bytes));
    workspace_.reset(ptr, [](void* ptr) { HIP_CALL_THROW(hipFree(ptr)); });
    params_.workspace_buffer = reinterpret_cast<T*>(workspace_.get());
  }

 protected:
  using ParamsT = contrib::rocm::GemmSoftmaxGemmPermuteParams<T>;
  rocblas_handle rocblas_handle_;
  hipDeviceProp_t device_prop;
  contrib::rocm::RocmAttentionParameters attn_;
  ParamsT params_;
  std::shared_ptr<void> workspace_;
};

// The pipeline composed from rocblas api calls and kernel launches.
template <typename T>
class GemmSoftmaxGemmPermuteGeneric : public IGemmSoftmaxGemmPermuteKernelExplorer<T> {
 public:
  GemmSoftmaxGemmPermuteGeneric(
      int64_t batch,
      int64_t seqlen,
      int64_t total_seqlen,
      std::optional<int64_t> max_seqlen,
      int64_t num_heads,
      int64_t head_size,
      int64_t mask_dim,
      double scale,
      contrib::AttentionQkvFormat qkv_format,
      DeviceArray& Q,
      std::optional<DeviceArray>& K,
      std::optional<DeviceArray>& V,
      std::optional<DeviceArray>& attn_bias,
      std::optional<DeviceArray>& attn_mask,
      DeviceArray& out)
      : IGemmSoftmaxGemmPermuteKernelExplorer<T>(batch, seqlen, total_seqlen, max_seqlen,
                                                 num_heads, head_size, mask_dim, scale, qkv_format,
                                                 Q, K, V, attn_bias, attn_mask, out) {
    this->SetWorkspace(GemmSoftmaxGemmPermuteGenericPipeline<T>::GetWorkspaceNumBytes(&this->attn_));
  }

  std::vector<std::string> ListOps() const {
    return {"Generic"};
  }

  bool SelectOp(const std::string&) {
    return true;
  }

  void Run() override {
    ORT_THROW_IF_ERROR(GemmSoftmaxGemmPermuteGenericPipeline<T>::Run(
        &this->params_, /*use_persistent_softmax=*/false));
  }
};

template <typename T>
class GemmSoftmaxGemmPermuteGenericNestedTunable : public GemmSoftmaxGemmPermuteGeneric<T> {
 public:
  GemmSoftmaxGemmPermuteGenericNestedTunable(
      int64_t batch,
      int64_t seqlen,
      int64_t total_seqlen,
      std::optional<int64_t> max_seqlen,
      int64_t num_heads,
      int64_t head_size,
      int64_t mask_dim,
      double scale,
      contrib::AttentionQkvFormat qkv_format,
      DeviceArray& Q,
      std::optional<DeviceArray>& K,
      std::optional<DeviceArray>& V,
      std::optional<DeviceArray>& attn_bias,
      std::optional<DeviceArray>& attn_mask,
      DeviceArray& out)
      : GemmSoftmaxGemmPermuteGeneric<T>(batch, seqlen, total_seqlen, max_seqlen,
                                         num_heads, head_size, mask_dim, scale, qkv_format,
                                         Q, K, V, attn_bias, attn_mask, out) {
    this->params_.TuningContext()->EnableTunableOpAndTuning();
  }
};

#ifdef USE_COMPOSABLE_KERNEL
template <typename T, bool USE_BIAS, bool USE_MASK>
class GemmSoftmaxGemmPermuteCK : public IGemmSoftmaxGemmPermuteKernelExplorer<T> {
 public:
  GemmSoftmaxGemmPermuteCK(
      int64_t batch,
      int64_t seqlen,
      int64_t total_seqlen,
      std::optional<int64_t> max_seqlen,
      int64_t num_heads,
      int64_t head_size,
      int64_t mask_dim,
      double scale,
      contrib::AttentionQkvFormat qkv_format,
      DeviceArray& Q,
      std::optional<DeviceArray>& K,
      std::optional<DeviceArray>& V,
      std::optional<DeviceArray>& attn_bias,
      std::optional<DeviceArray>& attn_mask,
      DeviceArray& out)
      : IGemmSoftmaxGemmPermuteKernelExplorer<T>(batch, seqlen, total_seqlen, max_seqlen,
                                                 num_heads, head_size, mask_dim, scale, qkv_format,
                                                 Q, K, V, attn_bias, attn_mask, out) {
    this->SetWorkspace(GemmSoftmaxGemmPermuteTunableOp<T>::GetWorkspaceNumBytes(&this->attn_));

    for (auto&& [ts, op] : GetCKGemmSoftmaxGemmPermuteTypeStringAndOps<T, USE_BIAS, USE_MASK>()) {
      type_strings_.emplace_back(std::move(ts));
      ops_.emplace_back(std::move(op));
    }
  }

  std::vector<std::string> ListOps() const {
    return type_strings_;
  }

  bool SelectOp(const std::string& name) {
    for (size_t i = 0; i < ops_.size(); i++) {
      if (type_strings_[i] == name) {
        selected_op_ = i;
        Status status = ops_[i].IsSupported(&this->params_);
        return status.IsOK();
      }
    }

    ORT_THROW("Cannot find implementation ", name);
  }

  void Run() override {
    ORT_THROW_IF_ERROR(ops_[selected_op_](&this->params_));
  }

 private:
  using ParamsT = typename IGemmSoftmaxGemmPermuteKernelExplorer<T>::ParamsT;
  using OpT = Op<ParamsT>;

  std::vector<OpT> ops_;
  std::vector<std::string> type_strings_;
  size_t selected_op_{};
};
#endif  // USE_COMPOSABLE_KERNEL

// The pipeline composed from rocblas api calls and kernel launches.
template <typename T>
class GemmSoftmaxGemmPermuteTunable : public IGemmSoftmaxGemmPermuteKernelExplorer<T> {
 public:
  GemmSoftmaxGemmPermuteTunable(
      int64_t batch,
      int64_t seqlen,
      int64_t total_seqlen,
      std::optional<int64_t> max_seqlen,
      int64_t num_heads,
      int64_t head_size,
      int64_t mask_dim,
      double scale,
      contrib::AttentionQkvFormat qkv_format,
      DeviceArray& Q,
      std::optional<DeviceArray>& K,
      std::optional<DeviceArray>& V,
      std::optional<DeviceArray>& attn_bias,
      std::optional<DeviceArray>& attn_mask,
      DeviceArray& out)
      : IGemmSoftmaxGemmPermuteKernelExplorer<T>(batch, seqlen, total_seqlen, max_seqlen,
                                                 num_heads, head_size, mask_dim, scale, qkv_format,
                                                 Q, K, V, attn_bias, attn_mask, out) {
    this->SetWorkspace(std::max(
        GemmSoftmaxGemmPermuteGenericPipeline<T>::GetWorkspaceNumBytes(&this->attn_),
        GemmSoftmaxGemmPermuteTunableOp<T>::GetWorkspaceNumBytes(&this->attn_)));

    this->params_.TuningContext()->EnableTunableOpAndTuning();
  }

  std::vector<std::string> ListOps() const {
    return {"Tunable"};
  }

  bool SelectOp(const std::string&) {
    return true;
  }

  void Run() override {
    ORT_THROW_IF_ERROR(op_(&this->params_));
  }

  // NOTE: this op is expensive to construct
  GemmSoftmaxGemmPermuteTunableOp<T> op_{};
};

#define REGISTER_COMMON(name, type, ...)                                                          \
  py::class_<type<__VA_ARGS__>>(m, name)                                                          \
      .def(py::init<int64_t, int64_t, int64_t, std::optional<int64_t>, int64_t, int64_t, int64_t, \
                    float, contrib::AttentionQkvFormat,                                           \
                    DeviceArray&,                                                                 \
                    std::optional<DeviceArray>&,                                                  \
                    std::optional<DeviceArray>&,                                                  \
                    std::optional<DeviceArray>&,                                                  \
                    std::optional<DeviceArray>&,                                                  \
                    DeviceArray&>())                                                              \
      .def("SetRepeats", &type<__VA_ARGS__>::SetRepeats)                                          \
      .def("Run", &type<__VA_ARGS__>::Run)                                                        \
      .def("Profile", &type<__VA_ARGS__>::Profile)                                                \
      .def("ListOps", &type<__VA_ARGS__>::ListOps)                                                \
      .def("SelectOp", &type<__VA_ARGS__>::SelectOp);

#define REGISTER_GENERIC(dtype) \
  REGISTER_COMMON("GemmSoftmaxGemmPermuteGeneric_" #dtype, GemmSoftmaxGemmPermuteGeneric, dtype)

#define REGISTER_GENERIC_NESTEDTUNABLE(dtype) \
  REGISTER_COMMON("GemmSoftmaxGemmPermuteGenericNestedTunable_" #dtype, GemmSoftmaxGemmPermuteGenericNestedTunable, dtype)

#define REGISTER_CK(dtype, biased, masked, mask_bias_suffix) \
  REGISTER_COMMON(                                           \
      "GemmSoftmaxGemmPermuteCK" mask_bias_suffix "_" #dtype, GemmSoftmaxGemmPermuteCK, dtype, biased, masked)

#define REGISTER_TUNABLE(dtype) \
  REGISTER_COMMON("GemmSoftmaxGemmPermuteTunable_" #dtype, GemmSoftmaxGemmPermuteTunable, dtype)

KE_REGISTER(m) {
  auto qkv_format = m.def_submodule("qkv_format");
  py::enum_<contrib::AttentionQkvFormat>(qkv_format, "qkv_format")
      .value("Q_K_V_BNSH", contrib::AttentionQkvFormat::Q_K_V_BNSH, "")
      .value("Q_K_V_BSNH", contrib::AttentionQkvFormat::Q_K_V_BSNH, "")
      .value("QKV_BSN3H", contrib::AttentionQkvFormat::QKV_BSN3H, "")
      .value("Q_KV_BSNH_BSN2H", contrib::AttentionQkvFormat::Q_KV_BSNH_BSN2H, "")
      .export_values();

  REGISTER_GENERIC(half);
  REGISTER_GENERIC(float);
  REGISTER_GENERIC_NESTEDTUNABLE(half);
  REGISTER_GENERIC_NESTEDTUNABLE(float);

#ifdef USE_COMPOSABLE_KERNEL
  REGISTER_CK(half, false, false, "");
  REGISTER_CK(half, true, false, "Biased");
  REGISTER_CK(half, false, true, "Masked");
  REGISTER_CK(half, true, true, "BiasedMasked");
#endif

  REGISTER_TUNABLE(half);
}

}  // namespace onnxruntime
