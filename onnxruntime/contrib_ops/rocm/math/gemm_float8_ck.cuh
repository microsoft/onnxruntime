// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <utility>
#include <vector>

#if defined(USE_COMPOSABLE_KERNEL)

#include "core/providers/rocm/composable_kernel_common.h"

#include "ck/ck.hpp"
#include "ck/utility/functional3.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_splitk.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#endif

#if !defined(DISABLE_FLOAT8_TYPES)
#include "core/framework/float8.h"
#endif
#include "core/providers/rocm/tunable/gemm_common.h"

namespace onnxruntime {
namespace rocm {
namespace tunable {

#if defined(USE_COMPOSABLE_KERNEL) && !defined(DISABLE_FLOAT8_TYPES)
using F8 = ck::f8_t;
using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

template <typename... Ts>
constexpr bool always_false = false;

template <typename F8>
struct Scale {
  constexpr const static bool is_pack2_invocable = true;
  constexpr const static bool is_pack4_invocable = true;

  explicit Scale(float scale_value, const float* dev_scale_ptr) : scale_value_{scale_value}, dev_scale_ptr_{dev_scale_ptr} {}

  template <typename Y, typename X>
  __forceinline__ __host__ __device__ Y fast_type_convert(X x) const {
    static_assert(always_false<X>, "not implemented");
    (void)x;
  }

  template <>
  __forceinline__ __host__ __device__ ck::half_t fast_type_convert<ck::half_t, ck::f8_t>(ck::f8_t x) const {
    // https://github.com/ROCmSoftwarePlatform/triton/blob/0cc3f8b84a16892396f6e08a04991034d67e32b1/lib/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVM.cpp#L220-L233
    constexpr const uint16_t mask = 0x7fff;
    constexpr const uint16_t sign_mask = 0x8000;
    constexpr const uint16_t exp_compensate = []() {
      if constexpr (std::is_same_v<F8, Float8E4M3FN>) {
        return 0x2000;
      } else if constexpr (std::is_same_v<F8, Float8E4M3FNUZ>) {
        return 0x1c00;
      }
    }();

    uint8_t x_u8 = reinterpret_cast<uint8_t&>(x);
    uint16_t x_u16 = static_cast<uint16_t>(x_u8) << 8;
    uint16_t exp = (x_u16 & mask) >> 1;
    uint16_t y = (x_u16 & sign_mask) | (exp + exp_compensate);
    return reinterpret_cast<ck::half_t&>(y);
  }

  __forceinline__ __host__ __device__ void operator()(ck::half_t& y, const ck::f8_t& x) const {
    float scale = scale_value_ * (*dev_scale_ptr_);
    y = ck::type_convert<ck::half_t>(scale * fast_type_convert<ck::half_t>(x));
  }

  __forceinline__ __host__ __device__ void operator()(ck::half2_t& ys, const ck::f8x2_t& xs) const {
    float scale = scale_value_ * (*dev_scale_ptr_);
    constexpr const uint32_t mask = 0x7fff7fff;
    constexpr const uint32_t sign_mask = 0x80008000;
    constexpr const uint32_t exp_compensate = []() {
      if constexpr (std::is_same_v<F8, Float8E4M3FN>) {
        return 0x20002000;
      } else if constexpr (std::is_same_v<F8, Float8E4M3FNUZ>) {
        return 0x1c001c00;
      }
    }();

    const uchar2& x2_u8 = reinterpret_cast<const uchar2&>(xs);
    uchar4 x{0, x2_u8.x, 0, x2_u8.y};
    uint32_t x_u32 = reinterpret_cast<uint32_t&>(x);

    uint32_t exp = (x_u32 & mask) >> 1;
    uint32_t v = (x_u32 & sign_mask) | (exp + exp_compensate);
    ys = scale * reinterpret_cast<ck::half2_t&>(v);
  }

  __forceinline__ __host__ __device__ void operator()(ck::half4_t& ys, const ck::f8x4_t& xs) const {
    float scale = scale_value_ * (*dev_scale_ptr_);
    constexpr const uint32_t mask = 0x7fff7fff;
    constexpr const uint32_t sign_mask = 0x80008000;
    constexpr const uint32_t exp_compensate = []() {
      if constexpr (std::is_same_v<F8, Float8E4M3FN>) {
        return 0x20002000;
      } else if constexpr (std::is_same_v<F8, Float8E4M3FNUZ>) {
        return 0x1c001c00;
      }
    }();

    uint32_t xs_u32 = reinterpret_cast<const uint32_t&>(xs);
    uint32_t x_u32_0 = __byte_perm(xs_u32, 0, 0x1504);
    uint32_t x_u32_1 = __byte_perm(xs_u32, 0, 0x3726);
    uint32_t exp_0 = (x_u32_0 & mask) >> 1;
    uint32_t exp_1 = (x_u32_1 & mask) >> 1;
    uint32_t v_0 = (x_u32_0 & sign_mask) | (exp_0 + exp_compensate);
    uint32_t v_1 = (x_u32_1 & sign_mask) | (exp_1 + exp_compensate);
    uint64_t v = v_0 | uint64_t(v_1) << 32;
    ys = scale * reinterpret_cast<ck::half4_t&>(v);
  }

  float scale_value_;
  const float* const dev_scale_ptr_;
};
#endif

namespace blas {

template <typename TA, typename TB, typename TC>
struct GemmFloat8Params : tunable::OpParams {
  std::string Signature() const override {
    return MakeString(BlasOpToString(opa), BlasOpToString(opb), "_", m, "_", n, "_", k);
  }

  rocblas_handle handle;
  BlasOp opa;
  BlasOp opb;
  int64_t m;
  int64_t n;
  int64_t k;
  float scale_a{};
  const float* scale_a_dev{};
  const TA* a;
  int64_t lda;
  float scale_b{};
  const float* scale_b_dev{};
  const TB* b;
  int64_t ldb;
  TC* c;
  float scale_c{};
  const float* scale_c_dev{};
  int64_t ldc;
};

#if defined(USE_COMPOSABLE_KERNEL) && !defined(DISABLE_FLOAT8_TYPES)

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using Nop = ck::tensor_operation::element_wise::PassThrough;

void add_device_gemm_xdl_splitk_f8_f16_f16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGemmSplitK<
        Row, Row, Row, F8, F16, F16, Scale<Float8E4M3FN>, Nop, Nop>>>& instances);

void add_device_gemm_xdl_splitk_f8_f16_f16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGemmSplitK<
        Row, Row, Row, F8, F16, F16, Scale<Float8E4M3FNUZ>, Nop, Nop>>>& instances);

void add_device_gemm_xdl_splitk_f16_f8_f16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGemmSplitK<
        Row, Row, Row, F16, F8, F16, Nop, Scale<Float8E4M3FN>, Nop>>>& instances);

void add_device_gemm_xdl_splitk_f16_f8_f16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGemmSplitK<
        Row, Row, Row, F16, F8, F16, Nop, Scale<Float8E4M3FNUZ>, Nop>>>& instances);

void add_device_gemm_xdl_splitk_f16_f8_f16_mk_nk_mn_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGemmSplitK<
        Row, Col, Row, F16, F8, F16, Nop, Scale<Float8E4M3FN>, Nop>>>& instances);

void add_device_gemm_xdl_splitk_f16_f8_f16_mk_nk_mn_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGemmSplitK<
        Row, Col, Row, F16, F8, F16, Nop, Scale<Float8E4M3FNUZ>, Nop>>>& instances);

template <typename OrtT>
auto CreateOp(float scale, const float* dev_scale) {
  if constexpr (std::is_same_v<OrtT, Float8E4M3FN>) {
    return Scale<Float8E4M3FN>(scale, dev_scale);
  } else if constexpr (std::is_same_v<OrtT, Float8E4M3FNUZ>) {
    return Scale<Float8E4M3FNUZ>(scale, dev_scale);
  } else {
    return Nop{};
  }
}

template <typename TA, typename TB, typename TC, typename ALayout, typename BLayout>
auto GetCKF8SplitKGemmTypeStringAndOps() {
  using CKTA = typename CKDataTypeAdaptor<TA>::type;
  using CKTB = typename CKDataTypeAdaptor<TB>::type;
  using CKTC = typename CKDataTypeAdaptor<TC>::type;

  using OpA = std::conditional_t<std::is_same_v<CKTA, ck::f8_t>, Scale<TA>, Nop>;
  using OpB = std::conditional_t<std::is_same_v<CKTB, ck::f8_t>, Scale<TB>, Nop>;
  using OpC = std::conditional_t<std::is_same_v<CKTC, ck::f8_t>, Scale<TC>, Nop>;

  using DeviceGemm = ck::tensor_operation::device::DeviceGemmSplitK<
      ALayout, BLayout, Row,
      CKTA, CKTB, CKTC,
      OpA, OpB, OpC>;

  std::vector<std::pair<std::string, Op<GemmFloat8Params<TA, TB, TC>>>> ret;

  for (auto num_split : {1, 4, 16, 64}) {
    std::vector<std::unique_ptr<DeviceGemm>> instances{};
    if constexpr (std::is_same_v<CKTA, ck::f8_t> && std::is_same_v<CKTB, ck::half_t> && std::is_same_v<CKTC, ck::half_t> &&
                  std::is_same_v<ALayout, Row> && std::is_same_v<BLayout, Row>) {
      add_device_gemm_xdl_splitk_f8_f16_f16_mk_kn_mn_instances(instances);
    } else if constexpr (std::is_same_v<CKTA, ck::half_t> && std::is_same_v<CKTB, ck::f8_t> && std::is_same_v<CKTC, ck::half_t> &&
                         std::is_same_v<ALayout, Row> && std::is_same_v<BLayout, Row>) {
      add_device_gemm_xdl_splitk_f16_f8_f16_mk_kn_mn_instances(instances);
    } else if constexpr (std::is_same_v<CKTA, ck::half_t> && std::is_same_v<CKTB, ck::f8_t> && std::is_same_v<CKTC, ck::half_t> &&
                         std::is_same_v<ALayout, Row> && std::is_same_v<BLayout, Col>) {
      add_device_gemm_xdl_splitk_f16_f8_f16_mk_nk_mn_instances(instances);
    } else {
      static_assert(always_false<CKTA, CKTB, CKTC, ALayout, BLayout>, "no instances for the type combination");
      LOGS_DEFAULT(FATAL) << "no instances for the type combination";
    }
    for (auto&& impl : instances) {
      auto type_string = std::to_string(ret.size()) + "_" + impl->GetTypeString() + "_SplitK" + std::to_string(num_split);
      auto invoker = impl->MakeInvokerPointer();
      auto ck_gemm_op = [num_split, impl = std::move(impl), invoker = std::move(invoker)](const GemmFloat8Params<TA, TB, TC>* params) -> Status {
        OpA op_a = CreateOp<TA>(params->scale_a, params->scale_a_dev);
        OpB op_b = CreateOp<TB>(params->scale_b, params->scale_b_dev);
        OpC op_c = CreateOp<TC>(params->scale_c, params->scale_c_dev);

        auto arg = impl->MakeArgumentPointer(params->a, params->b, params->c,
                                             params->m, params->n, params->k,
                                             params->lda, params->ldb, params->ldc,
                                             op_a, op_b, op_c, num_split);
        TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(!impl->IsSupportedArgument(arg.get()),
                                                  impl->GetTypeString(), " does not support ", params->Signature());
        invoker->Run(arg.get(), StreamConfig{params->StreamHandle()});
        return Status::OK();
      };
      ret.emplace_back(std::make_pair(std::move(type_string), std::move(ck_gemm_op)));
    }
  }
  return ret;
}

#endif  // USE_COMPOSABLE_KERNEL

template <typename TA, typename TB, typename TC, BlasOp OpA, BlasOp OpB>
class GemmFloat8TunableOp : public TunableOp<GemmFloat8Params<TA, TB, TC>> {
 public:
  GemmFloat8TunableOp() {
#if defined(USE_COMPOSABLE_KERNEL) && !defined(DISABLE_FLOAT8_TYPES)
    using ALayout = std::conditional_t<OpA == BlasOp::NonTrans, Row, Col>;
    using BLayout = std::conditional_t<OpB == BlasOp::NonTrans, Row, Col>;
    for (auto&& [_, op] : GetCKF8SplitKGemmTypeStringAndOps<TA, TB, TC, ALayout, BLayout>()) {
      ORT_UNUSED_PARAMETER(_);
      this->RegisterOp(std::move(op));
    }
#else
    ORT_ENFORCE(false, "CK is required to support GemmFloat8 computing");
#endif  // USE_COMPOSABLE_KERNEL
  }
};

}  // namespace blas
}  // namespace tunable
}  // namespace rocm
}  // namespace onnxruntime
