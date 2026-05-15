// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/tensor/scatter_elements.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

Status ScatterElementsProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& indices = shader.AddInput("indices", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  const auto& updates = shader.AddInput("updates", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseShapeAndStride);

  // Helper lambda for atomic reduction operations
  auto atomic_reduction_snippet = [](ScatterElementsReduction reduction, const std::string& base_ptr, const std::string& offset_var, const std::string& value, const std::string& data_type) -> std::string {
    std::ostringstream ss;
    bool is_32_bit_integer = data_type == "i32" || data_type == "u32";
    bool is_unsigned_integer = data_type == "u32";
    bool is_float16 = data_type == "f16";

    std::ostringstream ss_float_start;
    if (is_float16) {
      // For f16, we use u32 atomics where each u32 stores 2 f16 values
      // offset_var is the f16 index, so we need to:
      // 1. Calculate u32_offset = offset_var / 2
      // 2. Determine which half: offset_var % 2
      // 3. Update the appropriate half
      ss_float_start << "    {\n"
                     << "      let u32_offset = " << offset_var << " / 2u;\n"
                     << "      let is_lower_half = (" << offset_var << " % 2u) == 0u;\n"
                     << "      var oldValue = 0u;\n"
                     << "      loop {\n"
                     << "        let oldVec = unpack2x16float(oldValue);\n"
                     << "        let oldF16 = f16(select(oldVec.y, oldVec.x, is_lower_half));\n"
                     << "        let newValueF16 = ";
    } else {
      ss_float_start << "    {\n"
                     << "      var oldValue = 0" << (is_unsigned_integer ? "u" : "") << ";\n"
                     << "      loop {\n"
                     << "        let newValueF32 = ";
    }

    std::ostringstream ss_float_end;
    if (is_float16) {
      ss_float_end << ";\n"
                   << "        let updatedVec = select(\n"
                   << "          vec2<f32>(oldVec.x, f32(newValueF16)),\n"
                   << "          vec2<f32>(f32(newValueF16), oldVec.y),\n"
                   << "          is_lower_half\n"
                   << "        );\n"
                   << "        let newValue = pack2x16float(updatedVec);\n"
                   << "        let res = atomicCompareExchangeWeak(&" << base_ptr << "[u32_offset], oldValue, newValue);\n"
                   << "        if res.exchanged {\n"
                   << "          break;\n"
                   << "        }\n"
                   << "        oldValue = res.old_value;\n"
                   << "      }\n"
                   << "    }\n";
    } else {
      ss_float_end << ";\n"
                   << "        let newValue = bitcast<" << (is_unsigned_integer ? "u32" : "i32") << ">(newValueF32);\n"
                   << "        let res = atomicCompareExchangeWeak(&" << base_ptr << "[" << offset_var << "], oldValue, newValue);\n"
                   << "        if res.exchanged {\n"
                   << "          break;\n"
                   << "        }\n"
                   << "        oldValue = res.old_value;\n"
                   << "      }\n"
                   << "    }\n";
    }

    switch (reduction) {
      case ScatterElementsReduction::Add:
        if (is_32_bit_integer) {
          ss << "    atomicAdd(&" << base_ptr << "[" << offset_var << "], bitcast<" << data_type << ">(" << value << "));\n";
        } else if (is_float16) {
          ss << ss_float_start.str() << "oldF16 + (" << value << ")" << ss_float_end.str();
        } else {
          ss << ss_float_start.str() << "bitcast<" << data_type << ">(oldValue) + (" << value << ")" << ss_float_end.str();
        }
        break;
      case ScatterElementsReduction::Mul:
        if (is_float16) {
          ss << ss_float_start.str() << "(oldF16 * (" << value << "))" << ss_float_end.str();
        } else {
          ss << ss_float_start.str() << "(bitcast<" << data_type << ">(oldValue) * (" << value << "))" << ss_float_end.str();
        }
        break;
      case ScatterElementsReduction::Min:
        if (is_32_bit_integer) {
          ss << "    atomicMin(&" << base_ptr << "[" << offset_var << "], bitcast<" << data_type << ">(" << value << "));\n";
        } else if (is_float16) {
          ss << ss_float_start.str() << "min(oldF16, (" << value << "))" << ss_float_end.str();
        } else {
          ss << ss_float_start.str() << "min(bitcast<" << data_type << ">(oldValue), (" << value << "))" << ss_float_end.str();
        }
        break;
      case ScatterElementsReduction::Max:
        if (is_32_bit_integer) {
          ss << "    atomicMax(&" << base_ptr << "[" << offset_var << "], bitcast<" << data_type << ">(" << value << "));\n";
        } else if (is_float16) {
          ss << ss_float_start.str() << "max(oldF16, (" << value << "))" << ss_float_end.str();
        } else {
          ss << ss_float_start.str() << "max(bitcast<" << data_type << ">(oldValue), (" << value << "))" << ss_float_end.str();
        }
        break;
      default:
        ORT_THROW("Unsupported reduction type: ", static_cast<int>(reduction));
    }
    return ss.str();
  };

  // Determine data type string for atomic operations
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
  } else if (data_type_ == DataTypeImpl::GetType<MLFloat16>()) {
    reducible = true;
    data_type_str = "f16";
  } else {
    data_type_str = "output_value_t";
  }

  if (reduction_ != ScatterElementsReduction::None && !reducible) {
    ORT_THROW("ScatterElements: Reduction is not supported for data type ", data_type_str);
  }

  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size");

  // Convert linear index to multi-dimensional indices using indices OffsetToIndices
  shader.MainFunctionBody() << "  // Calculate output indices from global_idx\n"
                            << "  let update_indices = " << indices.OffsetToIndices("global_idx") << ";\n";

  // Get the scatter index from indices tensor
  shader.MainFunctionBody() << "  // Get the scatter index\n"
                            << "  var idx = i32(" << indices.GetByOffset("global_idx") << ");\n";

  // Handle negative indices
  shader.MainFunctionBody() << "  // Handle negative indices\n"
                            << "  if (idx < 0) {\n"
                            << "    idx = idx + i32(uniforms.axis_dim_limit);\n"
                            << "  }\n";

  // Bounds checking
  shader.MainFunctionBody() << "  // Bounds checking\n"
                            << "  if (idx < 0 || idx >= i32(uniforms.axis_dim_limit)) {\n"
                            << "    return;\n"
                            << "  }\n";

  // Build output indices by replacing the axis dimension with the scatter index
  shader.MainFunctionBody() << "  // Build output indices\n"
                            << "  var output_indices = update_indices;\n"
                            << output.IndicesSet("output_indices", std::to_string(axis_), "u32(idx)") << ";\n";

  // Get update value and scatter
  shader.MainFunctionBody() << "  let update_value = " << updates.GetByOffset("global_idx") << ";\n";
  shader.MainFunctionBody() << "  let output_offset = " << output.IndicesToOffset("output_indices") << ";\n";

  // Handle reduction
  if (reduction_ == ScatterElementsReduction::None) {
    // Non-reduction path: use direct assignment
    shader.MainFunctionBody() << "  " << output.SetByOffset("output_offset", "update_value") << ";\n";
  } else {
    // Reduction path: use atomic operations
    shader.MainFunctionBody() << atomic_reduction_snippet(reduction_, "output", "output_offset", "update_value", data_type_str);
  }

  return Status::OK();
}

Status ScatterElements::ComputeInternal(ComputeContext& context) const {
  const Tensor* input = context.Input<Tensor>(0);
  const Tensor* indices = context.Input<Tensor>(1);
  const Tensor* updates = context.Input<Tensor>(2);

  const auto& input_shape = input->Shape();
  const auto& indices_shape = indices->Shape();
  const auto& updates_shape = updates->Shape();

  const int64_t input_rank = static_cast<int64_t>(input_shape.NumDimensions());
  const int64_t axis = axis_ < 0 ? axis_ + input_rank : axis_;

  // Validate axis
  ORT_RETURN_IF_NOT(axis >= 0 && axis < input_rank, "axis ", axis_, " is out of bounds for tensor of rank ", input_rank);

  // Validate shapes
  ORT_RETURN_IF_NOT(indices_shape.NumDimensions() == updates_shape.NumDimensions(),
                    "Indices and updates must have the same rank");

  for (size_t i = 0; i < indices_shape.NumDimensions(); ++i) {
    ORT_RETURN_IF_NOT(indices_shape[i] == updates_shape[i],
                      "Indices and updates dimensions must match at position ", i);
  }

  auto* output = context.Output(0, input_shape);

  // Copy input to output if not in-place
  const void* source = input->DataRaw();
  void* target = output->MutableDataRaw();
  if (target != source) {
    ORT_RETURN_IF_ERROR(Info().GetDataTransferManager().CopyTensor(*input, *output));
  }

  // Early return if indices/updates are empty
  if (indices_shape.Size() == 0) {
    return Status::OK();
  }

  const uint32_t output_size = onnxruntime::narrow<uint32_t>(indices_shape.Size());
  const uint32_t axis_dim_limit = onnxruntime::narrow<uint32_t>(input_shape[static_cast<size_t>(axis)]);

  MLDataType data_type = input->DataType();
  ScatterElementsProgram program(axis, reduction_, data_type);

  program
      .CacheHint(std::to_string(axis) + "_" + std::to_string(static_cast<uint32_t>(reduction_)))
      .AddInputs({{indices, ProgramTensorMetadataDependency::TypeAndRank},
                  {updates, ProgramTensorMetadataDependency::TypeAndRank}})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({output_size, axis_dim_limit});

  // Use atomic output if reduction is enabled and data type supports it
  // Note: f16 uses atomic<u32> for reductions (packing 2 f16 values per u32)
  if (reduction_ != ScatterElementsReduction::None &&
      (data_type == DataTypeImpl::GetType<float>() ||
       data_type == DataTypeImpl::GetType<MLFloat16>() ||
       data_type == DataTypeImpl::GetType<int32_t>() ||
       data_type == DataTypeImpl::GetType<uint32_t>())) {
    program.AddOutput({output, ProgramTensorMetadataDependency::TypeAndRank, ProgramOutput::Atomic});
  } else {
    program.AddOutput({output, ProgramTensorMetadataDependency::TypeAndRank});
  }

  return context.RunProgram(program);
}

// Register kernels for different opset versions

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    ScatterElements,
    kOnnxDomain,
    11,
    12,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedNumberTypes())
        .TypeConstraint("Tind", BuildKernelDefConstraintsFromTypeList<TypeList<int32_t, int64_t>>())
        .MayInplace(0, 0),
    ScatterElements);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    ScatterElements,
    kOnnxDomain,
    13,
    15,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedNumberTypes())
        .TypeConstraint("Tind", BuildKernelDefConstraintsFromTypeList<TypeList<int32_t, int64_t>>())
        .MayInplace(0, 0),
    ScatterElements);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    ScatterElements,
    kOnnxDomain,
    16,
    17,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedNumberTypes())
        .TypeConstraint("Tind", BuildKernelDefConstraintsFromTypeList<TypeList<int32_t, int64_t>>())
        .MayInplace(0, 0),
    ScatterElements);

ONNX_OPERATOR_KERNEL_EX(
    ScatterElements,
    kOnnxDomain,
    18,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedNumberTypes())
        .TypeConstraint("Tind", BuildKernelDefConstraintsFromTypeList<TypeList<int32_t, int64_t>>())
        .MayInplace(0, 0),
    ScatterElements);

}  // namespace webgpu
}  // namespace onnxruntime
