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
using namespace onnxruntime::rocm;
#include "contrib_ops/rocm/bert/group_query_attention_impl.cuh"

namespace onnxruntime {

template <typename T>
class IGroupQueryAttentionKernelExplorer : public IKernelExplorer {
 public:
  IGroupQueryAttentionKernelExplorer(
      int64_t batch,
      int64_t seqlen,
      int64_t total_seqlen,
      int64_t num_heads,
      int64_t head_size,
      int64_t num_kv_heads,
      double scale,
      contrib::AttentionQkvFormat qkv_format,
      DeviceArray& Q,
      std::optional<DeviceArray>& K,
      std::optional<DeviceArray>& V,
      DeviceArray& out) {
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
    attn_.v_hidden_size = num_kv_heads * head_size;
    attn_.v_head_size = attn_.head_size;      // Q,K,V hidden size must agree now
    attn_.num_heads = num_heads;
    attn_.kv_num_heads = num_kv_heads;
    attn_.is_unidirectional = true;
    attn_.past_present_share_buffer = false;
    attn_.do_rotary = false;
    attn_.mask_filter_value = -10000.0f;
    attn_.scale = scale;
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
    params_.attention = &attn_;
    params_.device_prop = &device_prop;
    params_.scale = scale;

    std::tie(params_.q_buffer, params_.k_buffer, params_.v_buffer) = ConvertToOffsetedBufferViews<T>(
        &attn_, Q.ptr(), K.has_value() ? K->ptr() : nullptr, V.has_value() ? V->ptr() : nullptr);

    params_.out_buffer = reinterpret_cast<T*>(out.ptr());
  }

  ~IGroupQueryAttentionKernelExplorer() {
  }

 protected:
  using ParamsT = contrib::rocm::GroupedQueryAttentionParams<T>;
  hipDeviceProp_t device_prop;
  contrib::rocm::RocmAttentionParameters attn_;
  ParamsT params_;
};

#ifdef USE_COMPOSABLE_KERNEL
template <typename T>
class GroupQueryAttentionCK : public IGroupQueryAttentionKernelExplorer<T> {
 public:
  GroupQueryAttentionCK(
      int64_t batch,
      int64_t seqlen,
      int64_t total_seqlen,
      int64_t num_heads,
      int64_t head_size,
      int64_t num_kv_heads,
      double scale,
      contrib::AttentionQkvFormat qkv_format,
      DeviceArray& Q,
      std::optional<DeviceArray>& K,
      std::optional<DeviceArray>& V,
      DeviceArray& out)
      : IGroupQueryAttentionKernelExplorer<T>(batch, seqlen, total_seqlen,
                                                 num_heads, head_size, num_kv_heads, scale, qkv_format,
                                                 Q, K, V, out) {
    // for (auto&& [ts, op] : GetCKGroupedQueryAttentionTypeStringAndOps<T, 1>()) {
    //   type_strings_[1].emplace_back(std::move(ts));
    //   ops_.emplace_back(std::move(op));
    // }
    // for (auto&& [ts, op] : GetCKGroupedQueryAttentionTypeStringAndOps<T, 2>()) {
    //   type_strings_[2].emplace_back(std::move(ts));
    //   ops_.emplace_back(std::move(op));
    // }
    // for (auto&& [ts, op] : GetCKGroupedQueryAttentionTypeStringAndOps<T, 4>()) {
    //   type_strings_[4].emplace_back(std::move(ts));
    //   ops_.emplace_back(std::move(op));
    // }
    // for (auto&& [ts, op] : GetCKGroupedQueryAttentionTypeStringAndOps<T, 8>()) {
    //   type_strings_[8].emplace_back(std::move(ts));
    //   ops_.emplace_back(std::move(op));
    // }
    // for (auto&& [ts, op] : GetCKGroupedQueryAttentionTypeStringAndOps<T, 16>()) {
    //   type_strings_[16].emplace_back(std::move(ts));
    //   ops_.emplace_back(std::move(op));
    // }
    for (auto&& [ts, op] : GetCKGroupedQueryAttentionTypeStringAndOps<T, 32>()) {
      type_strings_[32].emplace_back(std::move(ts));
      ops_.emplace_back(std::move(op));
    }
  }

  std::vector<std::string> ListOps() const {
    std::vector<std::string> type_strings;
    for (auto&& [kv_heads, type_strs] : type_strings_) {
      for (auto&& type_str : type_strs) {
        type_strings.emplace_back(type_str);
      }
    }
    return type_strings;
  }

  bool SelectOp(const std::string& name) {
    int kv_heads = this->attn_.kv_num_heads;
    if (type_strings_.find(kv_heads) == type_strings_.end()) {
      return false;
    }
    auto &type_strs = type_strings_.at(kv_heads);
    for (size_t i = 0; i < ops_.size(); i++) {
      std::cout << "check for name: " << name << " type_strs[i]: " << type_strs[i] << std::endl;
      if (type_strs[i] == name) {
        selected_op_ = i;
        Status status = ops_[i].IsSupported(&this->params_);
        if (!status.IsOK()) {
          ORT_THROW("Implementation ", name, " is not supported: ", status.ErrorMessage());
        } else {
          return true;
        }
      }
    }

    ORT_THROW("Cannot find implementation ", name);
  }

  void Run() override {
    ORT_THROW_IF_ERROR(ops_[selected_op_](&this->params_));
  }

 private:
  using ParamsT = typename IGroupQueryAttentionKernelExplorer<T>::ParamsT;
  using OpT = Op<ParamsT>;

  std::vector<OpT> ops_;
  std::map<int, std::vector<std::string>> type_strings_;
  size_t selected_op_{};
};
#endif  // USE_COMPOSABLE_KERNEL

// The pipeline composed from rocblas api calls and kernel launches.
template <typename T>
class GroupQueryAttentionTunable : public IGroupQueryAttentionKernelExplorer<T> {
 public:
  GroupQueryAttentionTunable(
      int64_t batch,
      int64_t seqlen,
      int64_t total_seqlen,
      int64_t num_heads,
      int64_t head_size,
      int64_t num_kv_heads,
      double scale,
      contrib::AttentionQkvFormat qkv_format,
      DeviceArray& Q,
      std::optional<DeviceArray>& K,
      std::optional<DeviceArray>& V,
      DeviceArray& out)
      : IGroupQueryAttentionKernelExplorer<T>(batch, seqlen, total_seqlen,
                                                 num_heads, head_size, num_kv_heads, scale, qkv_format,
                                                 Q, K, V, out) {
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
  GroupedQueryAttentionTunableOp<T> op_{};
};

#define REGISTER_COMMON(name, type, ...)                                                          \
  py::class_<type<__VA_ARGS__>>(m, name)                                                          \
      .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,                         \
                    float, contrib::AttentionQkvFormat,                                           \
                    DeviceArray&,                                                                 \
                    std::optional<DeviceArray>&,                                                  \
                    std::optional<DeviceArray>&,                                                  \
                    DeviceArray&>())                                                              \
      .def("SetRepeats", &type<__VA_ARGS__>::SetRepeats)                                          \
      .def("Run", &type<__VA_ARGS__>::Run)                                                        \
      .def("Profile", &type<__VA_ARGS__>::Profile)                                                \
      .def("ListOps", &type<__VA_ARGS__>::ListOps)                                                \
      .def("SelectOp", &type<__VA_ARGS__>::SelectOp);

#define REGISTER_CK(dtype) \
  REGISTER_COMMON(                                           \
      "GroupQueryAttentionCK_" #dtype, GroupQueryAttentionCK, dtype)

#define REGISTER_TUNABLE(dtype) \
  REGISTER_COMMON("GroupQueryAttentionTunable_" #dtype, GroupQueryAttentionTunable, dtype)

KE_REGISTER(m) {

#ifdef USE_COMPOSABLE_KERNEL
  REGISTER_CK(half);
#endif

  REGISTER_TUNABLE(half);
}

}  // namespace onnxruntime
