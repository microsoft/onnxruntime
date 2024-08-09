// SPDX-License-Identifier: MIT
// Modifications Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#ifdef USE_COMPOSABLE_KERNEL
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/tensor_specialization.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm_softmax_gemm_permute.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_softmax_gemm_permute_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "ck/utility/data_type.hpp"

namespace onnxruntime {
namespace contrib {
namespace rocm {
namespace internal {

using F16 = ck::half_t;
using F32 = float;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using MaskingSpecialization = ck::tensor_operation::device::MaskingSpecialization;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using ck::tensor_operation::device::DeviceBatchedGemmSoftmaxGemmPermute;               // the interface
using ck::tensor_operation::device::DeviceBatchedGemmSoftmaxGemmPermute_Xdl_CShuffle;  // the implementation

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;
static constexpr auto GemmPadded = ck::tensor_operation::device::GemmSpecialization::MNKOPadding;

static constexpr auto TensorDefault = ck::tensor_operation::device::TensorSpecialization::Default;

template <ck::index_t NumDimG,
          ck::index_t NumDimM,
          ck::index_t NumDimN,
          ck::index_t NumDimK,
          ck::index_t NumDimO,
          typename DT,     // A, B0, B1, C, CShuffle Datatype
          typename D0sDT,  // D0 (bias in A*B0+D0s) DataType
          typename AccDT,  // Accumulator Datatype
          typename D0Op,   // D0 DataType
          MaskingSpecialization MaskingSpec>
using device_batched_gemm_softmax_gemm_permute_instances =
    std::tuple<
        // clang-format off
        // #############################################|  NumDimG| NumDimM| NumDimN| NumDimK| NumDimO| AData| B0Data| B1Data| CData| Acc0BiasData| Acc1BiasData| AccData| CShuffle|           A|          B0|        Acc0|          B1|           C|           GEMM|   ATensorSpec|  B0TensorSpec|  B1TensorSpec|   CTensorSpec| NumGemmK| Block| Gemm01| Gemm0| Gemm0| Gemm1| Gemm1| AK1| BK1| B1K1| MPer| NPer| Gemm0| Gemm0| Gemm1|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  B0BlockTransfer| B0BlockTransfer| B0BlockTransfer| B0BlockTransfer| B0BlockTransfer| B0BlockTransfer| B0BlockLds|  B1BlockTransfer| B1BlockTransfer| B1BlockTransfer| B1BlockTransfer| B1BlockTransfer| B1BlockTransfer| B1BlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer| MaskingSpec|  D0s Bias|
        // #############################################|         |        |        |        |        |  Type|   Type|   Type|  Type|         Type|         Type|    Type| DataType| Elementwise| Elementwise| Elementwise| Elementwise| Elementwise| Specialization|              |              |              |              | Prefetch|  Size|   MPer|  NPer|  KPer|  NPer|  KPer|    |    |     |  XDL|  XDL|  MXdl|  NXdl|  NXdl|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|    ThreadCluster|   ThreadCluster|  SrcAccessOrder|    SrcVectorDim|       SrcScalar|       DstScalar|  AddExtraN|    ThreadCluster|   ThreadCluster|  SrcAccessOrder|    SrcVectorDim|       SrcScalar|       DstScalar|  AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|            | SrcScalar|
        // #############################################|         |        |        |        |        |      |       |       |      |             |             |        |         |   Operation|   Operation|   Operation|   Operation|   Operation|               |              |              |              |              |    Stage|      |  Block| Block| Block| Block| Block|    |    |     |     |     |   Per|   Per|   Per| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          |  Lengths_K0_N_K1|    ArrangeOrder|                |                |       PerVector|    PerVector_K1|           |  Lengths_K0_N_K1|    ArrangeOrder|                |                |       PerVector|    PerVector_K1|           |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|            | PerVector|
        // #############################################|         |        |        |        |        |      |       |       |      |             |             |        |         |            |            |            |            |            |               |              |              |              |              |         |      |       |      |      |      |      |    |    |     |     |     |  Wave|  Wave|  Wave|                |               |               |               |               |               |          |                 |                |                |                |                |                |           |                 |                |                |                |                |                |           |            |            |                             |                |            |          |
        DeviceBatchedGemmSoftmaxGemmPermute_Xdl_CShuffle<  NumDimG, NumDimM, NumDimN, NumDimK, NumDimO,    DT,     DT,     DT,    DT,        D0sDT,  ck::Tuple<>,   AccDT,       DT, PassThrough, PassThrough,        D0Op, PassThrough, PassThrough,     GemmPadded, TensorDefault, TensorDefault, TensorDefault, TensorDefault,        1,   256,    128,    64,    32,   128,    32,   8,   8,    2,   32,   32,     1,     2,     4,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,      S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,       true,     S< 8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           2,               S<1, 32, 1, 8>,               8, MaskingSpec,       1>,
        DeviceBatchedGemmSoftmaxGemmPermute_Xdl_CShuffle<  NumDimG, NumDimM, NumDimN, NumDimK, NumDimO,    DT,     DT,     DT,    DT,        D0sDT,  ck::Tuple<>,   AccDT,       DT, PassThrough, PassThrough,        D0Op, PassThrough, PassThrough,    GemmDefault, TensorDefault, TensorDefault, TensorDefault, TensorDefault,        1,   256,    256,   128,    32,    64,    32,   8,   8,    2,   32,   32,     2,     4,     2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,      S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,       true,     S<16, 16, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           2,               S<1, 32, 1, 8>,               8, MaskingSpec>,
        DeviceBatchedGemmSoftmaxGemmPermute_Xdl_CShuffle<  NumDimG, NumDimM, NumDimN, NumDimK, NumDimO,    DT,     DT,     DT,    DT,        D0sDT,  ck::Tuple<>,   AccDT,       DT, PassThrough, PassThrough,        D0Op, PassThrough, PassThrough,    GemmDefault, TensorDefault, TensorDefault, TensorDefault, TensorDefault,        1,   256,    256,   128,    32,   128,    32,   8,   8,    2,   32,   32,     2,     4,     4,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,      S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,       true,     S< 8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           2,               S<1, 32, 1, 8>,               8, MaskingSpec>,
#if ROCM_VERSION >= 50500
        DeviceBatchedGemmSoftmaxGemmPermute_Xdl_CShuffle<  NumDimG, NumDimM, NumDimN, NumDimK, NumDimO,    DT,     DT,     DT,    DT,        D0sDT,  ck::Tuple<>,   AccDT,       DT, PassThrough, PassThrough,        D0Op, PassThrough, PassThrough,    GemmDefault, TensorDefault, TensorDefault, TensorDefault, TensorDefault,        1,   256,    128,   256,    32,    64,    32,   8,   8,    2,   32,   32,     1,     8,     2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,      S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,       true,     S<16, 16, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           2,               S<1, 32, 1, 8>,               8, MaskingSpec>,
#endif
        DeviceBatchedGemmSoftmaxGemmPermute_Xdl_CShuffle<  NumDimG, NumDimM, NumDimN, NumDimK, NumDimO,    DT,     DT,     DT,    DT,        D0sDT,  ck::Tuple<>,   AccDT,       DT, PassThrough, PassThrough,        D0Op, PassThrough, PassThrough,    GemmDefault, TensorDefault, TensorDefault, TensorDefault, TensorDefault,        1,   256,    128,   256,    32,   128,    32,   8,   8,    2,   32,   32,     1,     8,     4,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,      S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,       true,     S< 8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           2,               S<1, 32, 1, 8>,               8, MaskingSpec>,
        DeviceBatchedGemmSoftmaxGemmPermute_Xdl_CShuffle<  NumDimG, NumDimM, NumDimN, NumDimK, NumDimO,    DT,     DT,     DT,    DT,        D0sDT,  ck::Tuple<>,   AccDT,       DT, PassThrough, PassThrough,        D0Op, PassThrough, PassThrough,    GemmDefault, TensorDefault, TensorDefault, TensorDefault, TensorDefault,        1,   256,    128,   128,    64,    64,    32,   8,   8,    2,   32,   32,     1,     4,     2,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,     false,      S<8, 32, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,      false,     S<16, 16, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           2,               S<1, 32, 1, 8>,               8, MaskingSpec>,
        DeviceBatchedGemmSoftmaxGemmPermute_Xdl_CShuffle<  NumDimG, NumDimM, NumDimN, NumDimK, NumDimO,    DT,     DT,     DT,    DT,        D0sDT,  ck::Tuple<>,   AccDT,       DT, PassThrough, PassThrough,        D0Op, PassThrough, PassThrough,    GemmDefault, TensorDefault, TensorDefault, TensorDefault, TensorDefault,        1,   256,    128,   128,    32,    64,    32,   8,   8,    2,   32,   32,     1,     4,     2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,      S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,       true,     S<16, 16, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           2,               S<1, 32, 1, 8>,               8, MaskingSpec>,
        DeviceBatchedGemmSoftmaxGemmPermute_Xdl_CShuffle<  NumDimG, NumDimM, NumDimN, NumDimK, NumDimO,    DT,     DT,     DT,    DT,        D0sDT,  ck::Tuple<>,   AccDT,       DT, PassThrough, PassThrough,        D0Op, PassThrough, PassThrough,    GemmDefault, TensorDefault, TensorDefault, TensorDefault, TensorDefault,        1,   256,    128,   128,    64,   128,    32,   8,   8,    2,   32,   32,     1,     4,     4,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,     false,      S<8, 32, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,      false,     S< 8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           2,               S<1, 32, 1, 8>,               8, MaskingSpec>,
        DeviceBatchedGemmSoftmaxGemmPermute_Xdl_CShuffle<  NumDimG, NumDimM, NumDimN, NumDimK, NumDimO,    DT,     DT,     DT,    DT,        D0sDT,  ck::Tuple<>,   AccDT,       DT, PassThrough, PassThrough,        D0Op, PassThrough, PassThrough,    GemmDefault, TensorDefault, TensorDefault, TensorDefault, TensorDefault,        1,   256,    128,   128,    32,   128,    32,   8,   8,    2,   32,   32,     1,     4,     4,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,      S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,       true,     S< 8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           2,               S<1, 32, 1, 8>,               8, MaskingSpec>,
        DeviceBatchedGemmSoftmaxGemmPermute_Xdl_CShuffle<  NumDimG, NumDimM, NumDimN, NumDimK, NumDimO,    DT,     DT,     DT,    DT,        D0sDT,  ck::Tuple<>,   AccDT,       DT, PassThrough, PassThrough,        D0Op, PassThrough, PassThrough,    GemmDefault, TensorDefault, TensorDefault, TensorDefault, TensorDefault,        1,   256,     64,   256,    32,   128,    32,   8,   8,    2,   16,   16,     1,    16,     8,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,      S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,       true,     S< 8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           8,               S<1, 16, 1,16>,               8, MaskingSpec>,
        DeviceBatchedGemmSoftmaxGemmPermute_Xdl_CShuffle<  NumDimG, NumDimM, NumDimN, NumDimK, NumDimO,    DT,     DT,     DT,    DT,        D0sDT,  ck::Tuple<>,   AccDT,       DT, PassThrough, PassThrough,        D0Op, PassThrough, PassThrough,    GemmDefault, TensorDefault, TensorDefault, TensorDefault, TensorDefault,        1,   256,     64,   256,    32,    64,    32,   8,   8,    2,   16,   16,     1,    16,     4,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,      S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,       true,     S<16, 16, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           4,               S<1, 32, 1, 8>,               8, MaskingSpec>,
        DeviceBatchedGemmSoftmaxGemmPermute_Xdl_CShuffle<  NumDimG, NumDimM, NumDimN, NumDimK, NumDimO,    DT,     DT,     DT,    DT,        D0sDT,  ck::Tuple<>,   AccDT,       DT, PassThrough, PassThrough,        D0Op, PassThrough, PassThrough,    GemmDefault, TensorDefault, TensorDefault, TensorDefault, TensorDefault,        1,   256,     64,   256,    64,   128,    32,   8,   8,    2,   16,   16,     1,    16,     8,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,      S<8, 32, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,       true,     S< 8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           8,               S<1, 16, 1,16>,               8, MaskingSpec>,
        DeviceBatchedGemmSoftmaxGemmPermute_Xdl_CShuffle<  NumDimG, NumDimM, NumDimN, NumDimK, NumDimO,    DT,     DT,     DT,    DT,        D0sDT,  ck::Tuple<>,   AccDT,       DT, PassThrough, PassThrough,        D0Op, PassThrough, PassThrough,    GemmDefault, TensorDefault, TensorDefault, TensorDefault, TensorDefault,        1,   256,     64,   256,    64,    64,    32,   8,   8,    2,   16,   16,     1,    16,     4,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,      S<8, 32, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,       true,     S<16, 16, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           4,               S<1, 32, 1, 8>,               8, MaskingSpec>,
        // Padded fallback kernel
        DeviceBatchedGemmSoftmaxGemmPermute_Xdl_CShuffle<  NumDimG, NumDimM, NumDimN, NumDimK, NumDimO,    DT,     DT,     DT,    DT,        D0sDT,  ck::Tuple<>,   AccDT,       DT, PassThrough, PassThrough,        D0Op, PassThrough, PassThrough,     GemmPadded, TensorDefault, TensorDefault, TensorDefault, TensorDefault,        1,   256,    128,   128,    64,   128,    32,   8,   8,    2,   32,   32,     1,     4,     4,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,     false,      S<8, 32, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,      false,     S< 8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           2,               S<1, 32, 1, 8>,               8, MaskingSpec,       1>,
        DeviceBatchedGemmSoftmaxGemmPermute_Xdl_CShuffle<  NumDimG, NumDimM, NumDimN, NumDimK, NumDimO,    DT,     DT,     DT,    DT,        D0sDT,  ck::Tuple<>,   AccDT,       DT, PassThrough, PassThrough,        D0Op, PassThrough, PassThrough,     GemmPadded, TensorDefault, TensorDefault, TensorDefault, TensorDefault,        1,   256,    128,   128,    64,   128,    32,   8,   8,    2,   32,   32,     1,     4,     4,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,     false,      S<8, 32, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,      false,     S< 8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           2,               S<1, 32, 1, 8>,               8, MaskingSpec>,
        DeviceBatchedGemmSoftmaxGemmPermute_Xdl_CShuffle<  NumDimG, NumDimM, NumDimN, NumDimK, NumDimO,    DT,     DT,     DT,    DT,        D0sDT,  ck::Tuple<>,   AccDT,       DT, PassThrough, PassThrough,        D0Op, PassThrough, PassThrough,     GemmPadded, TensorDefault, TensorDefault, TensorDefault, TensorDefault,        1,   256,    128,    64,    32,   128,    32,   8,   8,    2,   32,   32,     1,     2,     4,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,      S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,       true,     S< 8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           2,               S<1, 32, 1, 8>,               8, MaskingSpec>
        // clang-format on
        >;

struct PreSoftmaxAttentionScoreOp {
  PreSoftmaxAttentionScoreOp(float scale) : scale_(scale) {}

  // non-biased, non-masked
  __host__ __device__ void operator()(float& y, const float& x) const {
    y = scale_ * x;
  }

  // biased or converted masked
  __host__ __device__ void operator()(float& y, const float& x, const F16& bias) const {
    y = scale_ * x + ck::type_convert<float>(bias);
  }

  // biased and converted masked
  __host__ __device__ void operator()(float& y, const float& x, const F16& bias, const F16& converted_mask) const {
    y = scale_ * x + ck::type_convert<float>(bias) + ck::type_convert<float>(converted_mask);
  }

  float scale_;
};

// Use this function to gat implementation
template <typename DT, typename D0sDT, typename AccDT, typename D0Op, MaskingSpecialization MaskingSpec>
std::vector<std::unique_ptr<DeviceBatchedGemmSoftmaxGemmPermute<
    2, 1, 1, 1, 1,
    DT, DT, DT, DT, D0sDT, ck::Tuple<>,
    PassThrough, PassThrough, D0Op, PassThrough, PassThrough,
    MaskingSpec>>>
GetDeviceBatchedGemmSoftmaxGemmPermuteInstances() {
  return {};
}

// implemented in impl_{fp16,bf16}[_biased][_masked].cu
// fp16, non-biased, non-masked
template <>
std::vector<std::unique_ptr<DeviceBatchedGemmSoftmaxGemmPermute<
    2, 1, 1, 1, 1,
    F16, F16, F16, F16, ck::Tuple<>, ck::Tuple<>,
    PassThrough, PassThrough, PreSoftmaxAttentionScoreOp, PassThrough, PassThrough,
    MaskingSpecialization::MaskDisabled>>>
GetDeviceBatchedGemmSoftmaxGemmPermuteInstances<
    F16, ck::Tuple<>, F32, PreSoftmaxAttentionScoreOp, MaskingSpecialization::MaskDisabled>();

// fp16, biased, non-masked
template <>
std::vector<std::unique_ptr<DeviceBatchedGemmSoftmaxGemmPermute<
    2, 1, 1, 1, 1,
    F16, F16, F16, F16, ck::Tuple<F16>, ck::Tuple<>,
    PassThrough, PassThrough, PreSoftmaxAttentionScoreOp, PassThrough, PassThrough,
    MaskingSpecialization::MaskDisabled>>>
GetDeviceBatchedGemmSoftmaxGemmPermuteInstances<
    F16, ck::Tuple<F16>, F32, PreSoftmaxAttentionScoreOp, MaskingSpecialization::MaskDisabled>();

// fp16, biased, fp16 masked, basically, two bias
template <>
std::vector<std::unique_ptr<DeviceBatchedGemmSoftmaxGemmPermute<
    2, 1, 1, 1, 1,
    F16, F16, F16, F16, ck::Tuple<F16, F16>, ck::Tuple<>,
    PassThrough, PassThrough, PreSoftmaxAttentionScoreOp, PassThrough, PassThrough,
    MaskingSpecialization::MaskDisabled>>>
GetDeviceBatchedGemmSoftmaxGemmPermuteInstances<
    F16, ck::Tuple<F16, F16>, F32, PreSoftmaxAttentionScoreOp, MaskingSpecialization::MaskDisabled>();

template <>
std::vector<std::unique_ptr<DeviceBatchedGemmSoftmaxGemmPermute<
    2, 1, 1, 1, 1,
    F16, F16, F16, F16, ck::Tuple<>, ck::Tuple<>,
    PassThrough, PassThrough, PreSoftmaxAttentionScoreOp, PassThrough, PassThrough,
    MaskingSpecialization::MaskOutUpperTriangle>>>
GetDeviceBatchedGemmSoftmaxGemmPermuteInstances<
    F16, ck::Tuple<>, F32, PreSoftmaxAttentionScoreOp, MaskingSpecialization::MaskOutUpperTriangle>();

// fp16, biased, non-masked
template <>
std::vector<std::unique_ptr<DeviceBatchedGemmSoftmaxGemmPermute<
    2, 1, 1, 1, 1,
    F16, F16, F16, F16, ck::Tuple<F16>, ck::Tuple<>,
    PassThrough, PassThrough, PreSoftmaxAttentionScoreOp, PassThrough, PassThrough,
    MaskingSpecialization::MaskOutUpperTriangle>>>
GetDeviceBatchedGemmSoftmaxGemmPermuteInstances<
    F16, ck::Tuple<F16>, F32, PreSoftmaxAttentionScoreOp, MaskingSpecialization::MaskOutUpperTriangle>();

// fp16, biased, fp16 masked, basically, two bias
template <>
std::vector<std::unique_ptr<DeviceBatchedGemmSoftmaxGemmPermute<
    2, 1, 1, 1, 1,
    F16, F16, F16, F16, ck::Tuple<F16, F16>, ck::Tuple<>,
    PassThrough, PassThrough, PreSoftmaxAttentionScoreOp, PassThrough, PassThrough,
    MaskingSpecialization::MaskOutUpperTriangle>>>
GetDeviceBatchedGemmSoftmaxGemmPermuteInstances<
    F16, ck::Tuple<F16, F16>, F32, PreSoftmaxAttentionScoreOp, MaskingSpecialization::MaskOutUpperTriangle>();

}  // namespace internal
}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
#endif  // USE_COMPOSABLE_KERNEL
