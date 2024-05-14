// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cann/cann_common.h"

namespace onnxruntime {
namespace cann {

template <>
const MLFloat16 Constants<MLFloat16>::Zero = MLFloat16::FromBits(static_cast<uint16_t>(0));
template <>
const MLFloat16 Constants<MLFloat16>::One = MLFloat16::FromBits(static_cast<uint16_t>(0x3C00));

template <>
const float Constants<float>::Zero = 0;
template <>
const float Constants<float>::One = 1;

}  // namespace cann
}  // namespace onnxruntime
