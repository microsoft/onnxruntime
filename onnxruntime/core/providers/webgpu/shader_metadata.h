// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/shader_config.h"

namespace onnxruntime::webgpu {

// The same declaration defines generation-time accessors and common identity.
#define WEBGPU_COMMON_PROGRAM_FIELDS(F)                                                                              \
  F(uint32_t, WorkgroupSizeX, program_.WorkgroupSizeX() == 0 ? uint32_t(WORKGROUP_SIZE) : program_.WorkgroupSizeX()) \
  F(uint32_t, WorkgroupSizeY, program_.WorkgroupSizeY() == 0 ? 1u : program_.WorkgroupSizeY())                       \
  F(uint32_t, WorkgroupSizeZ, program_.WorkgroupSizeZ() == 0 ? 1u : program_.WorkgroupSizeZ())                       \
  F(uint32_t, SubgroupSize, program_.SubgroupSize())                                                                 \
  F(bool, IsIndirect, program_.IndirectDispatchTensor() != nullptr)                                                  \
  F(size_t, InputCount, program_.Inputs().size())                                                                    \
  F(size_t, OutputCount, program_.Outputs().size())

#define WEBGPU_METADATA_ACCESSOR(type, name, expression) \
  type name() const { return expression; }
#define WEBGPU_METADATA_ENCODE(type, name, expression) AppendConfigValue(key, name());

class ShaderProgramMetadata final {
 public:
  explicit ShaderProgramMetadata(const ProgramBase& program) : program_{program} {}
  WEBGPU_COMMON_PROGRAM_FIELDS(WEBGPU_METADATA_ACCESSOR)
  void AppendTo(std::string& key) const { WEBGPU_COMMON_PROGRAM_FIELDS(WEBGPU_METADATA_ENCODE) }

 private:
  const ProgramBase& program_;
};

#define WEBGPU_COMMON_TENSOR_FIELDS(F)                                                                                             \
  F(ProgramVariableDataType, Type, tensor_.var_type)                                                                               \
  F(int, ElementType, tensor_.tensor->GetElementType())                                                                            \
  F(size_t, Rank, Shape().NumDimensions())                                                                                         \
  F(bool, HasStaticShape, (tensor_.dependency & ProgramTensorMetadataDependency::Shape) == ProgramTensorMetadataDependency::Shape) \
  F(uint32_t, Segments, segments_)                                                                                                 \
  F(bool, IsBufferView, tensor_.is_buffer_view)

template <typename TensorInfo>
class ShaderTensorMetadata final {
 public:
  explicit ShaderTensorMetadata(const TensorInfo& tensor, uint32_t segments = 1, size_t owner = 0)
      : tensor_{tensor}, segments_{segments}, owner_{owner} {}
  WEBGPU_COMMON_TENSOR_FIELDS(WEBGPU_METADATA_ACCESSOR)
  void AppendTo(std::string& key) const {
    WEBGPU_COMMON_TENSOR_FIELDS(WEBGPU_METADATA_ENCODE)
    if (HasStaticShape()) AppendConfigValue(key, Shape().GetDims());
    if (IsBufferView())
      AppendConfigValue(key, owner_);
    else
      AppendConfigValue(key, tensor_.buffer_offset_in_elements);
    if constexpr (std::is_same_v<TensorInfo, ProgramOutput>) AppendConfigValue(key, tensor_.is_atomic);
  }

 private:
  const TensorShape& Shape() const {
    return tensor_.use_override_shape ? tensor_.override_shape : tensor_.tensor->Shape();
  }
  const TensorInfo& tensor_;
  uint32_t segments_;
  size_t owner_;
};

#undef WEBGPU_COMMON_PROGRAM_FIELDS
#undef WEBGPU_COMMON_TENSOR_FIELDS
#undef WEBGPU_METADATA_ACCESSOR
#undef WEBGPU_METADATA_ENCODE

}  // namespace onnxruntime::webgpu
