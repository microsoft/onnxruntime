// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;
using onnxruntime::webgpu::ComputeContext;

enum class MoEActivationType {
  Relu,
  Gelu,
  Silu,
  Identity,
  SwiGLU,

};

enum class MoEQuantType {
  None = 0,
  UINT4 = 1,
  UINT8 = 2,
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
