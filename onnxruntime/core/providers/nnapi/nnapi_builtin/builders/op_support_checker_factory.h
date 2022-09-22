// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/nnapi/nnapi_builtin/builders/op_support_checker.h"

namespace onnxruntime {
namespace nnapi {

/* // The reason we use macros to create OpBuilders is for easy exclusion in build if certain op(s) are not used
// such that we can reduce binary size.
// This is for multiple ops share the same OpSupportChecker, we only need create one for all of them
#define NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER(OP_TYPE, SUPPORT_CHECKER_NAME) \
  SUPPORT_CHECKER_NAME::CreateSharedOpSupportChecker(OP_TYPE, op_registrations);

// This is for ops with dedicated OpSupportChecker
#define NNAPI_EP_ADD_SINGLE_OP_SUPPORT_CHECKER(OP_TYPE, SUPPORT_CHECKER_NAME)                                 \
  do {                                                                                                        \
    op_registrations.support_checkers.push_back(std::make_unique<SUPPORT_CHECKER_NAME>());                    \
    op_registrations.op_support_checker_map.emplace(OP_TYPE, op_registrations.support_checkers.back().get()); \
  } while (0) */

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
