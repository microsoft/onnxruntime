// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "core/framework/tuning_context.h"
#include "core/providers/cuda/cuda_execution_provider_info.h"

namespace onnxruntime {

class CUDAExecutionProvider;

namespace cuda {
namespace tunable {

class CudaTuningResultsValidator : public TuningResultsValidator {
 public:
  CudaTuningResultsValidator(CUDAExecutionProvider* ep);

 protected:
  std::string GetOrtBuildConfig() const override;

  std::string GetDeviceModel() const;
  Status ValidateDeviceModel(const std::string& value) const;

 private:
  CUDAExecutionProvider* ep_;  // non-owning handle
};

class CudaTuningContext : public ITuningContext {
 public:
  explicit CudaTuningContext(CUDAExecutionProvider* ep, TunableOpInfo* info);

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

 private:
  TunableOpInfo* info_;  // non-owning handle
  TuningResultsManager manager_;
  CudaTuningResultsValidator validator_;
};

}  // namespace tunable
}  // namespace cuda
}  // namespace onnxruntime
