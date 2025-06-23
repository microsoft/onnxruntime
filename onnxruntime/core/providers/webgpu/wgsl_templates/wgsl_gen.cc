// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>

#include "core/providers/webgpu/wgsl_templates/wgsl_gen.h"

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/shader_variable.h"
#include "core/providers/webgpu/webgpu_utils.h"

namespace onnxruntime {
namespace webgpu {
namespace wgsl_gen {

#if ORT_WGSL_TEMPLATE == 1  // Use static generator

#if defined(INCLUDED_BY_WGSL_GEN_IMPL)
#error "macro INCLUDED_BY_WGSL_GEN_IMPL should not be defined yet."
#endif

#define INCLUDED_BY_WGSL_GEN_IMPL
#include "wgsl_template_gen/index_impl.h"
#undef INCLUDED_BY_WGSL_GEN_IMPL

#elif ORT_WGSL_TEMPLATE == 2  // Use dynamic generator

#error "Dynamic WGSL template generation is not implemented yet."

#endif

}  // namespace wgsl_gen
}  // namespace webgpu
}  // namespace onnxruntime
