// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <array>
#include <unordered_map>

#include "core/common/common.h"
#include "core/platform/ort_mutex.h"
#include "core/framework/allocator.h"
#include "core/framework/tuning_results.h"

namespace onnxruntime {

class IExecutionProvider;
class TuningResultsManager;
class TuningResultsValidator;

class ITuningContext {
 public:
  explicit ITuningContext(IExecutionProvider* ep) : ep_(ep) {}
  virtual ~ITuningContext() = default;

  virtual void EnableTunableOp() = 0;
  virtual void DisableTunableOp() = 0;
  virtual bool IsTunableOpEnabled() const = 0;

  virtual void EnableTuning() = 0;
  virtual void DisableTuning() = 0;
  virtual bool IsTuningEnabled() const = 0;

  virtual void SetMaxTuningDurationMs(int max_duration_ms) = 0;
  virtual int GetMaxTuningDurationMs() const = 0;
  virtual void EnableTunableOpAndTuning() final {
    EnableTunableOp();
    EnableTuning();
  }

  virtual void DisableTunableOpAndTuning() final {
    DisableTunableOp();
    DisableTuning();
  }

  virtual TuningResultsManager& GetTuningResultsManager() = 0;
  virtual const TuningResultsManager& GetTuningResultsManager() const = 0;

  virtual const TuningResultsValidator& GetTuningResultsValidator() const = 0;

  virtual TuningResults GetTuningResults() const;
  virtual Status LoadTuningResults(const TuningResults& tr);

  void RegisterAllocatorsView(const AllocatorMap* allocators) { allocators_ = allocators; }

 protected:
  IExecutionProvider* ep_;
  const AllocatorMap* allocators_;
};

class TuningResultsManager {
 public:
  TuningResultsManager() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TuningResultsManager);

  KernelMap Lookup(const std::string& op_signature) const;
  int Lookup(const std::string& op_signature, const std::string& params_signature) const;

  void Add(const std::string& op_signature, const std::string& params_signature, int best_id);
  void Delete(const std::string& op_signature, const std::string& params_signature);

  void Load(const std::unordered_map<std::string, KernelMap>& results_to_load);
  std::unordered_map<std::string, KernelMap> Dump() const;

  void DisjointMerge(const std::string& op_signature, const KernelMap& kernel_map);

  // Mainly for testing purpose
  void Clear();

 private:
  mutable OrtMutex lock_;
  std::unordered_map<std::string, KernelMap> results_;
};

class TuningResultsValidator {
 public:
  using GetFunc = std::function<std::string()>;
  using ValidateFunc = std::function<Status(const std::string&)>;
  using GetValidateFuncs = std::unordered_map<std::string, std::pair<GetFunc, ValidateFunc>>;

  TuningResultsValidator();
  virtual ~TuningResultsValidator() = default;

  std::unordered_map<std::string, std::string> GetAllValidators() const;
  Status ValidateAll(const std::unordered_map<std::string, std::string>& to_validate) const;

 protected:
  void RegisterValidator(const std::string& key, const GetFunc& gf, const ValidateFunc& vf);

  virtual std::string GetOrtVersion() const;
  virtual Status ValidateOrtVersion(const std::string& value) const;

  virtual std::string GetOrtGitCommit() const;
  virtual Status ValidateOrtGitCommit(const std::string& value) const;

  virtual std::string GetOrtBuildConfig() const;
  virtual Status ValidateOrtBuildConfig(const std::string& value) const;

 public:
  static constexpr const std::array mandatory_keys{"ORT_VERSION", "ORT_GIT_COMMIT", "ORT_BUILD_CONFIG"};

 private:
  GetValidateFuncs validators_;
};

}  // namespace onnxruntime
