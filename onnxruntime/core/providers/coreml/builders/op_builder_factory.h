// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "op_builder.h"

namespace onnxruntime {
namespace coreml {

struct OpBuilderRegistrations {
  std::vector<std::unique_ptr<IOpBuilder>> builders;
  std::unordered_map<std::string, const IOpBuilder*> op_builder_map;
};

// Get the lookup table with IOpBuilder delegates for different onnx operators
const std::unordered_map<std::string, const IOpBuilder*>& GetOpBuilders();

void CreateActivationOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateArgMaxOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateBatchNormalizationOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateBinaryOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateCastOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateClipOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateConcatOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateConvOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateDepthToSpaceOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateFlattenOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateGatherOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateGemmOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateLRNOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreatePadOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreatePoolOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateReductionOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateReshapeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateResizeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateShapeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateSliceOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateSoftmaxOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateSqueezeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateTransposeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateUnaryOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);

}  // namespace coreml
}  // namespace onnxruntime
