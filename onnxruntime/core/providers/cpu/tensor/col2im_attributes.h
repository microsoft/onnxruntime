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
  explicit Col2ImAttributes(const OpKernelInfo& info) {
    if (!info.GetAttrs("strides", strides).IsOK())
      ORT_ENFORCE(strides.empty());
    if (!info.GetAttrs("dilations", dilations).IsOK())
      ORT_ENFORCE(dilations.empty());
    if (!info.GetAttrs("pads", pads).IsOK())
      ORT_ENFORCE(pads.empty());
  }

  ~Col2ImAttributes() = default;

  TensorShapeVector pads;
  TensorShapeVector dilations;
  TensorShapeVector strides;
};

}  // namespace onnxruntime
