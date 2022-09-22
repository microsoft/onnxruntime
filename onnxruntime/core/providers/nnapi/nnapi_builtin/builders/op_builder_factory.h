// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/nnapi/nnapi_builtin/builders/op_builder.h"

namespace onnxruntime {
namespace nnapi {

/* // The reason we use macros to create OpBuilders is for easy exclusion in build if certain op(s) are not used
// such that we can reduce binary size.
// This is for multiple ops share the same OpBuilder, we only need create one for all of them
#define NNAPI_EP_ADD_SHARED_OP_BUILDER(OP_TYPE, BUILDER_NAME) \
  BUILDER_NAME::CreateSharedOpBuilder(OP_TYPE, OpBuilderRegistrations& op_registrations);

// This is for ops with dedicated OpBuilder
#define NNAPI_EP_ADD_SINGLE_OP_BUILDER(OP_TYPE, BUILDER_NAME)                                 \
  do {                                                                                        \
    OpBuilderRegistrations& op_registrations.builders.push_back(std::make_unique<BUILDER_NAME>());                    \
    OpBuilderRegistrations& op_registrations.op_builder_map.emplace(OP_TYPE, OpBuilderRegistrations& op_registrations.builders.back().get()); \
  } while (0) */

struct OpBuilderRegistrations {
  std::vector<std::unique_ptr<IOpBuilder>> builders;
  std::unordered_map<std::string, const IOpBuilder*> op_builder_map;
};

const std::unordered_map<std::string, const IOpBuilder*>& GetOpBuilders();

void CreateBatchNormalizationOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateCastOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateClipOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateConcatOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateDepthToSpaceOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateDequantizeLinearOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateEluOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateFlattenOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateGatherOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateIdentityOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateLRNOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreatePadOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateQuantizeLinearOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateReluOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateReshapeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateResizeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateSliceOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateSoftMaxOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateSqueezeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateTransposeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateUnsqueezeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);

void CreateBinaryOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateConvOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateGemmOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreatePoolOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateMinMaxOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateUnaryOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);

}  // namespace nnapi
}  // namespace onnxruntime
