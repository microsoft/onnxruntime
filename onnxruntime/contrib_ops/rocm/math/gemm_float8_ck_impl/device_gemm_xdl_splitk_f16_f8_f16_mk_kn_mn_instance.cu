// SPDX-License-Identifier: MIT
// Modifications Copyright (c) Microsoft.
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_splitk_c_shuffle.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

#include "contrib_ops/rocm/math/gemm_float8_ck.cuh"

namespace onnxruntime {
namespace rocm {
namespace tunable {
namespace blas {
namespace internal {

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto GemmMNPadding = ck::tensor_operation::device::GemmSpecialization::MNPadding;

using ck::tensor_operation::device::DeviceGemmXdlSplitKCShuffle;

// The derived version is simply double BBlockTransferSrcScalarPerVector and adjust other values correspondingly
template <typename ScaleElemT>
using device_gemm_xdl_splitk_f16_f8_f16_mk_kn_mn_instances_ort = std::tuple<
    // clang-format off
        //#########################|AData| BData| CData| AccData| ALayout| BLayout| CLayout|           A|                 B|           C|          GEMM| Block|  MPer|  NPer| K0Per| K1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle|     CBlockTransferClusterLengths|  CBlockTransfer| Compute|
        //#########################| Type|  Type|  Type|    Type|        |        |        | Elementwise|       Elementwise| Elementwise|Specialization|  Size| Block| Block| Block|   |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave| _MBlock_MXdlPerWave_MWaveMPerXdl| ScalarPerVector|    Type|
        //#########################|     |      |      |        |        |        |        |   Operation|         Operation|   Operation|              |      |      |      |      |   |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle| _NBlock_NXdlPerWave_NWaveNPerXdl|   _NWaveNPerXdl|        |
        //#########################|     |      |      |        |        |        |        |            |                  |            |              |      |      |      |      |   |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                                 |                |        |
        DeviceGemmXdlSplitKCShuffle<  F16,    F8,   F16,     F32,     Row,      Row,    Row, PassThrough, Scale<ScaleElemT>, PassThrough, GemmMNPadding,   256,   256,   128,     8,  4,   32,   32,    4,    2,  S<1, 8, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              4,              4,      true,  S<1, 8, 32, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             2,              4,              4,      true,           1,           1,                   S<1, 32, 1, 8>,               8,    F16>,
        DeviceGemmXdlSplitKCShuffle<  F16,    F8,   F16,     F32,     Row,      Row,    Row, PassThrough, Scale<ScaleElemT>, PassThrough, GemmMNPadding,   256,   128,   256,     8,  4,   32,   32,    2,    4,  S<1, 8, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              4,              4,      true,  S<1, 8, 32, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             2,              8,              4,      true,           1,           1,                   S<1, 32, 1, 8>,               8,    F16>,
        DeviceGemmXdlSplitKCShuffle<  F16,    F8,   F16,     F32,     Row,      Row,    Row, PassThrough, Scale<ScaleElemT>, PassThrough, GemmMNPadding,   128,   128,   128,     8,  4,   32,   32,    4,    2,  S<1, 8, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              4,              4,      true,  S<1, 8, 16, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             2,              8,              4,      true,           1,           1,                   S<1, 16, 1, 8>,               8,    F16>,
        DeviceGemmXdlSplitKCShuffle<  F16,    F8,   F16,     F32,     Row,      Row,    Row, PassThrough, Scale<ScaleElemT>, PassThrough, GemmMNPadding,   256,    64,   192,     8,  4,   32,   32,    1,    3,  S<1, 8, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              4,              4,      true,  S<1, 8, 24, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             2,              4,              4,      true,           1,           1,                   S<1, 32, 1, 8>,               8,    F16>,
        DeviceGemmXdlSplitKCShuffle<  F16,    F8,   F16,     F32,     Row,      Row,    Row, PassThrough, Scale<ScaleElemT>, PassThrough, GemmMNPadding,   256,   192,    64,     8,  4,   32,   32,    3,    1,  S<1, 8, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              4,              4,      true,  S<1, 8, 16, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             2,              4,              4,      true,           1,           1,                   S<1, 32, 1, 8>,               8,    F16>,
        DeviceGemmXdlSplitKCShuffle<  F16,    F8,   F16,     F32,     Row,      Row,    Row, PassThrough, Scale<ScaleElemT>, PassThrough, GemmMNPadding,   256,   128,   128,     8,  4,   32,   32,    2,    2,  S<1, 8, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              4,              4,      true,  S<1, 8, 32, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             2,              4,              4,      true,           1,           1,                   S<1, 32, 1, 8>,               8,    F16>,
        DeviceGemmXdlSplitKCShuffle<  F16,    F8,   F16,     F32,     Row,      Row,    Row, PassThrough, Scale<ScaleElemT>, PassThrough, GemmMNPadding,   128,   128,    64,     8,  4,   32,   32,    2,    2,  S<1, 8, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              4,              4,      true,  S<1, 8, 16, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             2,              4,              4,      true,           1,           1,                   S<1, 32, 1, 4>,               8,    F16>,
        DeviceGemmXdlSplitKCShuffle<  F16,    F8,   F16,     F32,     Row,      Row,    Row, PassThrough, Scale<ScaleElemT>, PassThrough, GemmMNPadding,   128,    64,   128,     8,  4,   32,   32,    2,    2,  S<1, 8, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              4,              4,      true,  S<1, 8, 16, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             2,              8,              4,      true,           1,           1,                   S<1, 16, 1, 8>,               8,    F16>,
        DeviceGemmXdlSplitKCShuffle<  F16,    F8,   F16,     F32,     Row,      Row,    Row, PassThrough, Scale<ScaleElemT>, PassThrough, GemmMNPadding,   256,   128,    64,     8,  4,   32,   32,    2,    1,  S<1, 8, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              4,              4,      true,  S<1, 8, 32, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             2,              2,              4,      true,           1,           1,                   S<1, 16, 1, 4>,               8,    F16>,
        DeviceGemmXdlSplitKCShuffle<  F16,    F8,   F16,     F32,     Row,      Row,    Row, PassThrough, Scale<ScaleElemT>, PassThrough, GemmMNPadding,   256,    64,   128,     8,  4,   32,   32,    1,    2,  S<1, 8, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              4,              4,      true,  S<1, 8, 32, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             2,              4,              4,      true,           1,           1,                   S<1, 32, 1, 8>,               8,    F16>,
        DeviceGemmXdlSplitKCShuffle<  F16,    F8,   F16,     F32,     Row,      Row,    Row, PassThrough, Scale<ScaleElemT>, PassThrough, GemmMNPadding,   128,    32,   192,     8,  4,   32,   32,    1,    3,  S<1, 8, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              4,              4,      true,  S<1, 8, 12, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             2,             16,              4,      true,           1,           1,                   S<1, 16, 1, 8>,               8,    F16>,
        DeviceGemmXdlSplitKCShuffle<  F16,    F8,   F16,     F32,     Row,      Row,    Row, PassThrough, Scale<ScaleElemT>, PassThrough, GemmMNPadding,   128,   192,    32,     8,  4,   32,   32,    3,    1,  S<1, 8, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              4,              4,      true,  S<1, 8, 16, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             2,              2,              4,      true,           1,           1,                   S<1, 32, 1, 4>,               8,    F16>,
        DeviceGemmXdlSplitKCShuffle<  F16,    F8,   F16,     F32,     Row,      Row,    Row, PassThrough, Scale<ScaleElemT>, PassThrough, GemmMNPadding,   128,    32,    64,     8,  4,   32,   32,    1,    1,  S<1, 8, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              4,              4,      true,  S<1, 8, 16, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             2,              4,              4,      true,           1,           1,                   S<1, 16, 1, 8>,               8,    F16>,
        DeviceGemmXdlSplitKCShuffle<  F16,    F8,   F16,     F32,     Row,      Row,    Row, PassThrough, Scale<ScaleElemT>, PassThrough, GemmMNPadding,   128,    64,    32,     8,  4,   32,   32,    1,    1,  S<1, 8, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              4,              4,      true,  S<1, 8, 16, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             2,              2,              4,      true,           1,           1,                   S<1, 32, 1, 4>,               8,    F16>,
        DeviceGemmXdlSplitKCShuffle<  F16,    F8,   F16,     F32,     Row,      Row,    Row, PassThrough, Scale<ScaleElemT>, PassThrough, GemmMNPadding,   128,    32,   128,     8,  4,   32,   32,    1,    2,  S<1, 8, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              4,              4,      true,  S<1, 8, 16, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             2,              8,              4,      true,           1,           1,                   S<1, 16, 1, 8>,               8,    F16>,
        DeviceGemmXdlSplitKCShuffle<  F16,    F8,   F16,     F32,     Row,      Row,    Row, PassThrough, Scale<ScaleElemT>, PassThrough, GemmMNPadding,   128,   128,    32,     8,  4,   32,   32,    2,    1,  S<1, 8, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              4,              4,      true,  S<1, 8, 16, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             2,              2,              4,      true,           1,           1,                   S<1, 32, 1, 4>,               8,    F16>
    // clang-format on
    >;

void add_device_gemm_xdl_splitk_f16_f8_f16_mk_kn_mn_instances_ort(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGemmSplitK<
        Row, Row, Row, F16, F8, F16, PassThrough, Scale<Float8E4M3FN>, PassThrough>>>&
        instances) {
  ck::tensor_operation::device::instance::add_device_operation_instances(
      instances,
      device_gemm_xdl_splitk_f16_f8_f16_mk_kn_mn_instances_ort<Float8E4M3FN>{});
}

void add_device_gemm_xdl_splitk_f16_f8_f16_mk_kn_mn_instances_ort(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGemmSplitK<
        Row, Row, Row, F16, F8, F16, PassThrough, Scale<Float8E4M3FNUZ>, PassThrough>>>&
        instances) {
  ck::tensor_operation::device::instance::add_device_operation_instances(
      instances,
      device_gemm_xdl_splitk_f16_f8_f16_mk_kn_mn_instances_ort<Float8E4M3FNUZ>{});
}

}  // namespace internal
}  // namespace blas
}  // namespace tunable
}  // namespace rocm
}  // namespace onnxruntime
