// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/**
 * @file wgsl_gen.cc
 * @brief WGSL shader generation for WebGPU execution provider
 *
 * This file implements WGSL template generation using pre-compiled C++
 * template functions to generate optimized WGSL shader code.
 */

#include <string>

#include "core/providers/webgpu/wgsl_templates/wgsl_gen.h"

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/shader_variable.h"
#include "core/providers/webgpu/webgpu_utils.h"

namespace onnxruntime {
namespace webgpu {
namespace wgsl_gen {

#if defined(INCLUDED_BY_WGSL_GEN_IMPL)
#error "macro INCLUDED_BY_WGSL_GEN_IMPL should not be defined yet."
#endif

#define INCLUDED_BY_WGSL_GEN_IMPL
#include "wgsl_template_gen/index_impl.h"
#undef INCLUDED_BY_WGSL_GEN_IMPL

}  // namespace wgsl_gen
}  // namespace webgpu
}  // namespace onnxruntime
