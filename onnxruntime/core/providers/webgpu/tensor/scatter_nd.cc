// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <iostream>
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/shader_helper.h"
#include "scatter_nd.h"

namespace onnxruntime {
namespace webgpu {

Status ScatterNDProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("indices", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  shader.AddInput("updates", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseShapeAndStride);
  const auto output_rank = output.Rank();
  auto atomic_reduction_snippet = [](ScatterNDReduction reduction, const std::string& ptr, const std::string& value, const std::string& data_type) -> std ::string {
    std::ostringstream ss;
    bool is_32_bit_integer = data_type == "i32" || data_type == "u32";

    std::ostringstream ss_float_start;
    ss_float_start << "    {\n"
                   << "      var oldValue = 0;\n"
                   << "      loop {\n"
                   << "        let newValueF32 = ";
    std::ostringstream ss_float_end;
    ss_float_end << ";\n"
                 << "          let newValue = bitcast<i32>(newValueF32);\n"
                 << "          let res = atomicCompareExchangeWeak(&" << ptr << ", oldValue, newValue);\n"
                 << "          if res.exchanged {\n"
                 << "            break;\n"
                 << "          }\n"
                 << "          oldValue = res.old_value;\n"
                 << "        }\n"
                 << "      }\n";
    switch (reduction) {
      case ScatterNDReduction::None:
        ss << "  " << ptr << " = " << value << ";\n";
        break;
      case ScatterNDReduction::Add:
        if (is_32_bit_integer) {
          ss << "  atomicAdd(&" << ptr << ", bitcat<" << data_type << ">(" << value << "));\n";
        } else {
          // atomicAdd only supports uint/int type. For float, we use
          // atomicCompareExchangeWeak to simulate.
          ss << ss_float_start.str() << "bitcast<" << data_type << ">(oldValue) + (" << value << ")" << ss_float_end.str()
             << "\n";
        }
        break;
      case ScatterNDReduction::Max:
        if (is_32_bit_integer) {
          ss << "  atomicMax(&" << ptr << ", bitcat<" << data_type << ">(" << value << "));\n";
        } else {
          // atomicAdd only supports uint/int type. For float, we use
          // atomicCompareExchangeWeak to simulate.
          ss << ss_float_start.str() << "max(bitcast<" << data_type << ">(oldValue), (" << value << "))" << ss_float_end.str();
        }
        break;
      case ScatterNDReduction::Min:
        if (is_32_bit_integer) {
          ss << "  atomicMin(&" << ptr << ", bitcat<" << data_type << ">(" << value << "));\n";
        } else {
          // atomicAdd only supports uint/int type. For float, we use
          // atomicCompareExchangeWeak to simulate.
          ss << ss_float_start.str() << "min(bitcast<" << data_type << ">(oldValue), (" << value << "))" << ss_float_end.str();
        }
        break;
      case ScatterNDReduction::Mul:
        // atomicAdd only supports uint/int type. For float, we use
        // atomicCompareExchangeWeak to simulate.
        ss << ss_float_start.str() << "(bitcast<" << data_type << ">(oldValue) * (" << value << "))" << ss_float_end.str();
        break;
      default:;
        // The controlflow should never reach here.
    }
    return ss.str();
  };

  auto calc_data_offset_snippet = [](uint32_t output_rank, bool parallel) -> std::string {
    std::ostringstream ss;
    if (output_rank <= 2) {
      ss << "    let element_count_dim = uniforms.output_stride;\n";
    } else {
      ss << "    let element_count_dim = uniforms.output_stride[" << (parallel ? "i - indices_start" : "i") << "];\n";
    }
    if (output_rank == 1) {
      ss << "    let dim_value = uniforms.output_shape;\n";
    } else {
      ss << "    let dim_value = uniforms.output_shape[" << (parallel ? "i - indices_start" : "i") << " + uniforms.last_index_dimension];\n";
    }

    ss << "    if (index >= 0) {\n"
       << "      if (index >= i32(dim_value)) {\n"
       << "        index = i32(dim_value - 1);\n"
       << "      }\n"
       << "    } else {\n"
       << "      if (index < -i32(dim_value)) {\n"
       << "        index = 0;\n"
       << "      } else {\n"
       << "        index += i32(dim_value);\n"
       << "      }\n"
       << "    }\n"
       << "    data_offset += u32((u32(index) * element_count_dim));\n";
    return ss.str();
  };

  auto update_elements_snippet = [atomic_reduction_snippet](ScatterNDReduction reduction, const std::string& data_type, bool parallel) -> std::string {
    std::ostringstream ss;
    ss << "  for (var i = 0u; i < uniforms.num_updates_elements; i++) {\n"
       << "    let value = updates[uniforms.num_updates_elements * " << (parallel ? "global_idx" : "idx") << " + i];\n"
       << atomic_reduction_snippet(reduction, "output[data_offset + i]", "value", data_type) << "\n"
       << "  }\n";
    return ss.str();
  };
  std::string data_type_str;
  bool reducible = false;
  if (data_type_ == DataTypeImpl::GetType<int32_t>()) {
    reducible = true;
    data_type_str = "i32";
  } else if (data_type_ == DataTypeImpl::GetType<uint32_t>()) {
    reducible = true;
    data_type_str = "u32";
  } else if (data_type_ == DataTypeImpl::GetType<float>()) {
    reducible = true;
    data_type_str = "f32";
  } else {
    // Default value.
    data_type_str = "output_element_t";
  }
  if (reduction_ != ScatterNDReduction::None && !reducible) {
    ORT_THROW("ScatterND: Reduction is not supported for data type ", data_type_str);
  }
  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
                            << "  var hasDuplicates = false;\n";
  if (reduction_ == ScatterNDReduction::None) {
    shader.MainFunctionBody() << "  for (var i = 0u; i < uniforms.num_indices_elements; i = i + 1) {\n"
                              << "    for (var j = i + 1; j < uniforms.num_indices_elements; j = j + 1) {\n"
                              << "      var index_i = i32(indices[i].x);\n"
                              << "      var index_j = i32(indices[j].x);\n"
                              << "      if (index_i == index_j) {\n"
                              << "        hasDuplicates = true;\n"
                              << "        break;\n"
                              << "      }\n"
                              << "    }\n"
                              << "    if (hasDuplicates) {\n"
                              << "      break;\n"
                              << "    }\n"
                              << "  }\n"
                              << "\n"
                              << "  if (hasDuplicates) {\n"
                              << "    if (global_idx != 0u) {\n"
                              << "      return;\n"
                              << "    }\n"
                              << "    // Process each index-update pair individually when duplicates exist\n"
                              << "    for (var idx = 0u; idx < uniforms.num_indices_elements; idx++) {\n"
                              << "      var data_offset = 0u;\n"
                              << "      for (var i = 0u; i < uniforms.last_index_dimension; i++) {\n"
                              << "        var index = i32(indices[idx * uniforms.last_index_dimension + i].x);\n"
                              << calc_data_offset_snippet(output_rank, false)
                              << "      }\n"
                              << update_elements_snippet(reduction_, data_type_str, false)
                              << "    }\n"
                              << "    return;\n"
                              << "  }\n";
  }
  shader.MainFunctionBody() << "  var data_offset = 0u;\n"
                            << "  var indices_start = uniforms.last_index_dimension * global_idx;\n"
                            << "  var indices_end = indices_start + uniforms.last_index_dimension;\n"
                            << "  for (var i = indices_start; i < indices_end; i++) {\n"
                            << "    var index = i32(indices[i].x);\n"
                            << calc_data_offset_snippet(output_rank, true)
                            << "  }\n"
                            << update_elements_snippet(reduction_, data_type_str, true);
  return Status::OK();
}

Status ScatterND::ComputeInternal(ComputeContext& context) const {
  const Tensor* input = context.Input<Tensor>(0);
  const auto* indices = context.Input<Tensor>(1);
  const auto* updates = context.Input<Tensor>(2);
  const auto& input_shape = input->Shape();
  const auto& indices_shape = indices->Shape();
  auto indices_rank = indices_shape.NumDimensions();
  auto last_index_dimension = static_cast<uint32_t>(indices_shape[indices_rank - 1]);
  auto num_updates_elements = static_cast<uint32_t>(input_shape.SizeFromDimension(last_index_dimension));
  auto num_indices_elements = static_cast<uint32_t>(indices_shape.SizeToDimension(indices_rank - 2));
  // TODO: support bool with components 4.
  const size_t components = 1;
  auto output_size = static_cast<uint32_t>((indices_shape.Size() + components - 1) / components);
  auto* output = context.Output(0, input_shape);
  MLDataType data_type = input->DataType();
  const void* source = input->DataRaw();
  void* target = output->MutableDataRaw();
  // If source and target pointers are not equal (non-inplace operation), we need to copy the data.
  if (target != source) {
    ORT_RETURN_IF_ERROR(Info().GetDataTransferManager().CopyTensor(*input, *output));
  }
  ScatterNDProgram program(reduction_, data_type);
  program
      .CacheHint(static_cast<uint32_t>(reduction_))
      .AddInputs({{indices, ProgramTensorMetadataDependency::TypeAndRank},
                  {updates, ProgramTensorMetadataDependency::TypeAndRank}})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({output_size, last_index_dimension, num_indices_elements, num_updates_elements});
  if (reduction_ != ScatterNDReduction::None) {
    program.AddOutput({output, ProgramTensorMetadataDependency::TypeAndRank, ProgramOutput::Atomic});
  } else {
    program.AddOutput({output, ProgramTensorMetadataDependency::TypeAndRank});
  }
  return context.RunProgram(program);
}

ONNX_OPERATOR_KERNEL_EX(
    ScatterND,
    kOnnxDomain,
    18,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedNumberTypes())
        .MayInplace(0, 0),
        ScatterND);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    ScatterND,
    kOnnxDomain,
    16,
    17,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedNumberTypes())
        .MayInplace(0, 0),
        ScatterND);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    ScatterND,
    kOnnxDomain,
    13,
    15,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedNumberTypes())
        .MayInplace(0, 0),
        ScatterND);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    ScatterND,
    kOnnxDomain,
    11,
    12,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedNumberTypes())
        .MayInplace(0, 0),
        ScatterND);
}  // namespace webgpu
}  // namespace onnxruntime
