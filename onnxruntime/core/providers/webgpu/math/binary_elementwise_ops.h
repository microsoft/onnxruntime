// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/kernel_registry.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"

namespace onnxruntime {
namespace webgpu {

enum class BinaryImplementation { None,
                                  Min,
                                  Max,
                                  Pow };

#define WEBGPU_BINARY_ELEMENTWISE_PROGRAM_CONFIG(F) \
  F(std::string, program_name_)                     \
  F(ShaderLiteral, expression_)                     \
  F(BinaryImplementation, implementation_)          \
  F(bool, is_broadcast_)                            \
  F(bool, is_lhs_scalar_)                           \
  F(bool, is_rhs_scalar_)                           \
  F(bool, is_lhs_use_4_components_)                 \
  F(bool, is_rhs_use_4_components_)                 \
  F(bool, vectorize_)                               \
  F(bool, is_int64_input_)                          \
  F(bool, is_int64_output_)

struct BinaryElementwiseProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_BINARY_ELEMENTWISE_PROGRAM_CONFIG);
    Config(const std::string& kernel_name, ShaderLiteral expression, BinaryImplementation implementation,
           const bool is_broadcast, const bool is_lhs_scalar, const bool is_rhs_scalar,
           const bool is_lhs_use_4_components, const bool is_rhs_use_4_components, const bool vectorize,
           const bool is_int64_input = false, const bool is_int64_output = false)
        : program_name_{kernel_name},
          expression_{expression},
          implementation_{implementation},
          is_broadcast_{is_broadcast},
          is_lhs_scalar_{is_lhs_scalar},
          is_rhs_scalar_{is_rhs_scalar},
          is_lhs_use_4_components_{is_lhs_use_4_components},
          is_rhs_use_4_components_{is_rhs_use_4_components},
          vectorize_{vectorize},
          is_int64_input_{is_int64_input},
          is_int64_output_{is_int64_output} {}
  };
  static std::string_view Name(const Config& config) { return config.program_name_; }
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"vec_size", ProgramUniformVariableDataType::Uint32},
                                          {"element_count", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_BINARY_ELEMENTWISE_PROGRAM_CONFIG

using BinaryElementwiseProgram = ConfiguredProgram<BinaryElementwiseProgramShader>;

class BinaryElementwise : public WebGpuKernel {
 public:
  BinaryElementwise(const OpKernelInfo& info,
                    const std::string& kernel_name,
                    ShaderLiteral expression,
                    BinaryImplementation implementation = BinaryImplementation::None) : WebGpuKernel{info},
                                                                                        kernel_name_{kernel_name},
                                                                                        expression_{expression},
                                                                                        implementation_{implementation} {}

 protected:
  Status ComputeInternal(ComputeContext& context) const final;

 private:
  std::string kernel_name_;
  ShaderLiteral expression_;
  const BinaryImplementation implementation_;
};

// Registers the binary elementwise ops (Add, Sub, Mul, Div, Max, Min, Equal, Greater, Less,
// GreaterOrEqual, LessOrEqual, Pow, PRelu, And) through a single path. int64 support (behind the
// enableInt64 provider option) is enabled for the arithmetic/comparison ops whose i32 shader
// semantics are meaningful; Pow int64 is a known gap (TODO) and PRelu/And have no int64 form.
void RegisterBinaryElementwiseKernels(KernelRegistry& kernel_registry, bool enable_int64);

// Variadic element-wise operator (e.g. Max, Min) that accepts 1..N inputs with
// multidirectional (NumPy-style) broadcasting. The inputs are folded pairwise using the
// two-input binary element-wise program, reusing its broadcasting and vectorization paths.
class VariadicElementwise : public WebGpuKernel {
 public:
  VariadicElementwise(const OpKernelInfo& info,
                      const std::string& kernel_name,
                      ShaderLiteral expression,
                      BinaryImplementation implementation = BinaryImplementation::None) : WebGpuKernel{info},
                                                                                          kernel_name_{kernel_name},
                                                                                          expression_{expression},
                                                                                          implementation_{implementation} {}

 protected:
  Status ComputeInternal(ComputeContext& context) const final;

 private:
  std::string kernel_name_;
  ShaderLiteral expression_;
  const BinaryImplementation implementation_;
};

}  // namespace webgpu
}  // namespace onnxruntime
