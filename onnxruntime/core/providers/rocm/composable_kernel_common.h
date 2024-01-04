// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef USE_COMPOSABLE_KERNEL
#include "ck/utility/data_type.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#endif

#include "core/framework/float8.h"
#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/tunable/gemm_common.h"

namespace onnxruntime {
namespace rocm {

#ifdef USE_COMPOSABLE_KERNEL
template <tunable::blas::BlasOp Op>
struct CKBlasOpAdaptor {
  using type = std::conditional_t<Op == tunable::blas::BlasOp::NonTrans,
                                  ck::tensor_layout::gemm::RowMajor,
                                  ck::tensor_layout::gemm::ColumnMajor>;
};

template <typename T>
struct CKDataTypeAdaptor {
  using type = T;
};

template <>
struct CKDataTypeAdaptor<half> {
  using type = ck::half_t;
};

template <>
struct CKDataTypeAdaptor<MLFloat16> {
  using type = ck::half_t;
};

template <>
struct CKDataTypeAdaptor<BFloat16> {
  using type = ck::bhalf16_t;
};

#if !defined(DISABLE_FLOAT8_TYPES)
template <>
struct CKDataTypeAdaptor<Float8E4M3FN> {
  using type = ck::f8_t;
};

template <>
struct CKDataTypeAdaptor<Float8E4M3FNUZ> {
  using type = ck::f8_t;
};
#endif

#endif

}  // namespace rocm
}  // namespace onnxruntime
