// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/nnapi/nnapi_builtin/builders/op_support_checker.h"

namespace onnxruntime {
namespace nnapi {

struct OpSupportCheckerRegistrations {
  std::vector<std::unique_ptr<IOpSupportChecker>> support_checkers;
  std::unordered_map<std::string, const IOpSupportChecker*> op_support_checker_map;
};

const std::unordered_map<std::string, const IOpSupportChecker*>& GetOpSupportCheckers();

void CreateBatchNormalizationOpSupportChecker(const std::string& op_type, OpSupportCheckerRegistrations& op_registrations);
void CreateCastOpSupportChecker(const std::string& op_type, OpSupportCheckerRegistrations& op_registrations);
void CreateClipOpSupportChecker(const std::string& op_type, OpSupportCheckerRegistrations& op_registrations);
void CreateConcatOpSupportChecker(const std::string& op_type, OpSupportCheckerRegistrations& op_registrations);
void CreateDepthToSpaceOpSupportChecker(const std::string& op_type, OpSupportCheckerRegistrations& op_registrations);
void CreateDequantizeLinearOpSupportChecker(const std::string& op_type, OpSupportCheckerRegistrations& op_registrations);
void CreateEluOpSupportChecker(const std::string& op_type, OpSupportCheckerRegistrations& op_registrations);
void CreateFlattenOpSupportChecker(const std::string& op_type, OpSupportCheckerRegistrations& op_registrations);
void CreateGatherOpSupportChecker(const std::string& op_type, OpSupportCheckerRegistrations& op_registrations);
void CreateLRNOpSupportChecker(const std::string& op_type, OpSupportCheckerRegistrations& op_registrations);
void CreatePadOpSupportChecker(const std::string& op_type, OpSupportCheckerRegistrations& op_registrations);
void CreateQuantizeLinearOpSupportChecker(const std::string& op_type, OpSupportCheckerRegistrations& op_registrations);
void CreateReshapeOpSupportChecker(const std::string& op_type, OpSupportCheckerRegistrations& op_registrations);
void CreateResizeOpSupportChecker(const std::string& op_type, OpSupportCheckerRegistrations& op_registrations);
void CreateSliceOpSupportChecker(const std::string& op_type, OpSupportCheckerRegistrations& op_registrations);
void CreateSoftMaxOpSupportChecker(const std::string& op_type, OpSupportCheckerRegistrations& op_registrations);
void CreateSqueezeOpSupportChecker(const std::string& op_type, OpSupportCheckerRegistrations& op_registrations);
void CreateTransposeOpSupportChecker(const std::string& op_type, OpSupportCheckerRegistrations& op_registrations);
void CreateUnsqueezeOpSupportChecker(const std::string& op_type, OpSupportCheckerRegistrations& op_registrations);

void CreateBaseOpSupportChecker(const std::string& op_type, OpSupportCheckerRegistrations& op_registrations);
void CreateBinaryOpSupportChecker(const std::string& op_type, OpSupportCheckerRegistrations& op_registrations);
void CreateConvOpSupportChecker(const std::string& op_type, OpSupportCheckerRegistrations& op_registrations);
void CreateGemmOpSupportChecker(const std::string& op_type, OpSupportCheckerRegistrations& op_registrations);
void CreateMinMaxOpSupportChecker(const std::string& op_type, OpSupportCheckerRegistrations& op_registrations);
void CreatePoolOpSupportChecker(const std::string& op_type, OpSupportCheckerRegistrations& op_registrations);
void CreateUnaryOpSupportChecker(const std::string& op_type, OpSupportCheckerRegistrations& op_registrations);

}  // namespace nnapi
}  // namespace onnxruntime
