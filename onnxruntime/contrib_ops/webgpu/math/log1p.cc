// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/math/unary_elementwise_ops.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"

namespace onnxruntime::contrib::webgpu {
namespace {

// Evaluate the small-input branch in f32, including for shader-f16 inputs.
// 2*atanh(x/(2+x)) avoids cancellation; |x| <= 0.5 implies |s| <= 1/3.
// For |x| <= 2^-25, log1p(x) rounds to x. Return x directly so a normal
// input near f32's minimum cannot flush to zero in the intermediate x/2.
constexpr const char* log1p_impl = R"(
fn log1p_v(a: x_value_t) -> x_value_t {
  let x = vec4<f32>(a);
  let small = abs(x) <= vec4<f32>(0.5);
  let t = select(vec4<f32>(0.0), x, small);
  let s = t / (2.0 + t);
  let z = s * s;
  let p = 1.0 + z * (1.0/3.0 + z * (1.0/5.0 + z * (1.0/7.0 +
          z * (1.0/9.0 + z * (1.0/11.0 + z * (1.0/13.0 +
          z * (1.0/15.0 + z * (1.0/17.0))))))));
  let result = select(log(1.0 + x), 2.0 * s * p, small);
  return x_value_t(select(result, x, abs(x) <= vec4<f32>(2.98023223876953125e-8)));
}
)";

class Log1p final : public onnxruntime::webgpu::UnaryElementwise {
 public:
  Log1p(const OpKernelInfo& info)
      : UnaryElementwise(info, "Log1p", "log1p_v(a)", log1p_impl,
                         onnxruntime::webgpu::ShaderUsage::UseValueTypeAlias) {}
};

}  // namespace

ONNX_OPERATOR_KERNEL_EX(
    Log1p, kMSDomain, 1, kWebGpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", onnxruntime::webgpu::WebGpuSupportedFloatTypes()), Log1p);

}  // namespace onnxruntime::contrib::webgpu
