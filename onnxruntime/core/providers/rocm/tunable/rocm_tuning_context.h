// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "core/framework/tuning_context.h"
#include "core/providers/rocm/rocm_execution_provider_info.h"

namespace onnxruntime {

class ROCMExecutionProvider;

namespace rocm {
namespace tunable {

class RocmTuningResultsValidator : public TuningResultsValidator {
 public:
  RocmTuningResultsValidator(ROCMExecutionProvider* ep);

 protected:
  std::string GetOrtBuildConfig() const override;

  std::string GetDeviceModel() const;
  Status ValidateDeviceModel(const std::string& value) const;

 private:
  ROCMExecutionProvider* ep_;  // non-owning handle
};

class RocmTuningContext : public ITuningContext {
 public:
  explicit RocmTuningContext(ROCMExecutionProvider* ep, TunableOpInfo* info);

  void EnableTunableOp() override;
  void DisableTunableOp() override;
  bool IsTunableOpEnabled() const override;

  void EnableTuning() override;
  void DisableTuning() override;
  bool IsTuningEnabled() const override;

  void SetMaxTuningDurationMs(int max_duration_ms) override;
  int GetMaxTuningDurationMs() const override;

  TuningResultsManager& GetTuningResultsManager() override;
  const TuningResultsManager& GetTuningResultsManager() const override;

  const TuningResultsValidator& GetTuningResultsValidator() const override;

  IAllocatorUniquePtr<void> GetScratchBuffer(
      size_t bytes, Stream* stream, OrtMemType mem_type = OrtMemTypeDefault) const;

 private:
  TunableOpInfo* info_;  // non-owning handle
  TuningResultsManager manager_;
  RocmTuningResultsValidator validator_;
};

}  // namespace tunable
}  // namespace rocm
}  // namespace onnxruntime
