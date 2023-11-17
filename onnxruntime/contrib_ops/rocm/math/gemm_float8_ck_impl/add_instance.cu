// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_splitk_c_shuffle.hpp"

#include "contrib_ops/rocm/math/gemm_float8_ck.cuh"

namespace onnxruntime {
namespace rocm {
namespace tunable {
namespace blas {
namespace internal {

using F8 = ck::f8_t;
using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

void add_device_gemm_xdl_splitk_f16_f8_f16_mk_kn_mn_instances_ck(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGemmSplitK<
        Row, Row, Row, F16, F8, F16, PassThrough, Scale<Float8E4M3FN>, PassThrough>>>& instances);

void add_device_gemm_xdl_splitk_f16_f8_f16_mk_kn_mn_instances_ck(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGemmSplitK<
        Row, Row, Row, F16, F8, F16, PassThrough, Scale<Float8E4M3FNUZ>, PassThrough>>>& instances);

void add_device_gemm_xdl_splitk_f16_f8_f16_mk_kn_mn_instances_ort(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGemmSplitK<
        Row, Row, Row, F16, F8, F16, PassThrough, Scale<Float8E4M3FN>, PassThrough>>>& instances);

void add_device_gemm_xdl_splitk_f16_f8_f16_mk_kn_mn_instances_ort(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGemmSplitK<
        Row, Row, Row, F16, F8, F16, PassThrough, Scale<Float8E4M3FNUZ>, PassThrough>>>& instances);

void add_device_gemm_xdl_splitk_f16_f8_f16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGemmSplitK<
        Row, Row, Row, F16, F8, F16, PassThrough, Scale<Float8E4M3FN>, PassThrough>>>&
        instances) {
  add_device_gemm_xdl_splitk_f16_f8_f16_mk_kn_mn_instances_ck(instances);
  add_device_gemm_xdl_splitk_f16_f8_f16_mk_kn_mn_instances_ort(instances);
}

void add_device_gemm_xdl_splitk_f16_f8_f16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGemmSplitK<
        Row, Row, Row, F16, F8, F16, PassThrough, Scale<Float8E4M3FNUZ>, PassThrough>>>&
        instances) {
  add_device_gemm_xdl_splitk_f16_f8_f16_mk_kn_mn_instances_ck(instances);
  add_device_gemm_xdl_splitk_f16_f8_f16_mk_kn_mn_instances_ort(instances);
}

void add_device_gemm_xdl_splitk_f8_f16_f16_mk_kn_mn_instances_ck(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGemmSplitK<
        Row, Row, Row, F8, F16, F16, Scale<Float8E4M3FN>, PassThrough, PassThrough>>>& instances);

void add_device_gemm_xdl_splitk_f8_f16_f16_mk_kn_mn_instances_ck(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGemmSplitK<
        Row, Row, Row, F8, F16, F16, Scale<Float8E4M3FNUZ>, PassThrough, PassThrough>>>& instances);

// TODO: The first try of derivation does not going well due to various constraints.
// void add_device_gemm_xdl_splitk_f8_f16_f16_mk_kn_mn_instances_ort(
//     std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGemmSplitK<
//         Row, Row, Row, F8, F16, F16, Scale<Float8E4M3FN>, PassThrough, PassThrough>>>& instances);

// TODO: The first try of derivation does not going well due to various constraints.
// void add_device_gemm_xdl_splitk_f8_f16_f16_mk_kn_mn_instances_ort(
//     std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGemmSplitK<
//         Row, Row, Row, F8, F16, F16, Scale<Float8E4M3FNUZ>, PassThrough, PassThrough>>>& instances);

void add_device_gemm_xdl_splitk_f8_f16_f16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGemmSplitK<
        Row, Row, Row, F8, F16, F16, Scale<Float8E4M3FN>, PassThrough, PassThrough>>>& instances) {
  add_device_gemm_xdl_splitk_f8_f16_f16_mk_kn_mn_instances_ck(instances);
  //   add_device_gemm_xdl_splitk_f8_f16_f16_mk_kn_mn_instances_ort(instances);  // TODO:
}

void add_device_gemm_xdl_splitk_f8_f16_f16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGemmSplitK<
        Row, Row, Row, F8, F16, F16, Scale<Float8E4M3FNUZ>, PassThrough, PassThrough>>>& instances) {
  add_device_gemm_xdl_splitk_f8_f16_f16_mk_kn_mn_instances_ck(instances);
  //   add_device_gemm_xdl_splitk_f8_f16_f16_mk_kn_mn_instances_ort(instances);  // TODO:
}

}  // namespace internal
}  // namespace blas
}  // namespace tunable
}  // namespace rocm
}  // namespace onnxruntime
