// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/js/js_kernel.h"
#include "core/providers/js/operators/conv_transpose.h"

namespace onnxruntime {
namespace contrib {
namespace js {

using onnxruntime::js::ConvTranspose;

template <bool is_channels_last>
class FusedConvTranspose : public ConvTranspose<is_channels_last> {
 public:
  explicit FusedConvTranspose(const OpKernelInfo& info) : ConvTranspose<is_channels_last>(info, true) {
  }
};

}  // namespace js
}  // namespace contrib
}  // namespace onnxruntime
