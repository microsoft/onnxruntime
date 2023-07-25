// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef USE_COMPOSABLE_KERNEL
#include "ck/utility/data_type.hpp"
#endif

#include "core/providers/rocm/rocm_common.h"

namespace onnxruntime {
namespace rocm {

#ifdef USE_COMPOSABLE_KERNEL
template <typename T>
struct CKDataTypeAdaptor {
  using type = T;
};

template <>
struct CKDataTypeAdaptor<half> {
  using type = ck::half_t;
};

template <>
struct CKDataTypeAdaptor<BFloat16> {
  using type = ck::bhalf16_t;
};
#endif

}  // namespace rocm
}  // namespace onnxruntime
