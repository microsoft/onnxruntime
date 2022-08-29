/**
* Copyright (c) 2016-present, Facebook, Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
/* Modifications Copyright (c) Microsoft. */

#pragma once

#include "core/common/common.h"
#include "core/providers/common.h"
#include "core/util/math.h"

#include "core/common/inlined_containers.h"
#include "core/framework/op_kernel.h"
#include "core/framework/op_node_proto_helper.h"

namespace onnxruntime {

struct Col2ImAttributes {
  using Col2ImPadVector = InlinedVector<int64_t, kTensorShapeSmallBufferElementsSize * 2>;

  explicit Col2ImAttributes(const OpKernelInfo& info) {
    // Make sure empty strides, pads or dilations are defaulted to 1 if necessary
    ORT_ENFORCE(info.GetAttrs("strides", strides).IsOK());
    gsl::span<const int64_t> pads_span;
    ORT_ENFORCE(info.GetAttrsAsSpan("pads", pads_span).IsOK());
    pads.assign(pads_span.cbegin(), pads_span.cend());
    ORT_ENFORCE(info.GetAttrs("dilations", dilations).IsOK());
  }

  ~Col2ImAttributes() = default;

  Col2ImPadVector pads;
  TensorShapeVector dilations;
  TensorShapeVector strides;
};

}  // namespace onnxruntime
