// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/framework/ort_value.h"
#include "core/session/kv_cache_manager.h"
#include "core/generate/token_generation.h"

namespace onnxruntime {

class InferenceSession;
class SessionLoopExecution;

enum class LoopExitKind { EndOfInputs,
                          MaxIterations,
                          MaxGeneratedTokens,
                          Eos,
                          BooleanOutput };

struct LoopExitCondition {
  LoopExitKind kind;
  int64_t limit = 0;
  std::string output_name;
  InlinedVector<int64_t> eos_token_ids;
};

struct LoopFeedbackBinding {
  std::string input_name;
  std::string output_name;
};

struct SessionLoopOptions {
  InlinedVector<std::string> finite_input_names;
  InlinedVector<LoopFeedbackBinding> feedback;
  InlinedVector<std::string> output_names;
  InlinedVector<LoopExitCondition> exit_conditions;
  // A scalar int64 model output represents one generated token per iteration.
  std::string generated_token_output;
  // Reject an empty finite source or empty tensor for this input (e.g. a prompt).
  std::string nonempty_input;
  InlinedVector<LoopKVCacheBinding, 1> kv_caches;
  std::optional<TokenGenerationOptions> generation;
  StatefulProviderSelection cache_provider;
  StatefulProviderSelection sampler_provider;
};

using LoopFeeds = InlinedHashMap<std::string, OrtValue>;

struct SessionLoopInputs {
  // Initial feedback values and invariant inputs; finite inputs belong in batches.
  LoopFeeds inputs;
  InlinedVector<LoopFeeds> batches;
};

struct SessionLoopStepResult {
  bool executed = false;
  bool completed = false;
  int64_t iterations = 0;
  int64_t generated_tokens = 0;
  InlinedVector<LoopExitKind> exit_reasons;
  // Owned CPU snapshots, in the definition's output_names order.
  InlinedVector<OrtValue> outputs;
  std::optional<int64_t> generated_token;
};

// The caller-owned, initialized Session must outlive the definition and executions.
class SessionLoop : public std::enable_shared_from_this<SessionLoop> {
 public:
  static Status Create(InferenceSession& session, SessionLoopOptions options,
                       std::shared_ptr<const SessionLoop>& out);
  Status CreateExecution(SessionLoopInputs inputs, std::unique_ptr<SessionLoopExecution>& out) const;
  const SessionLoopOptions& Options() const { return options_; }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SessionLoop);

 private:
  friend class SessionLoopExecution;
  friend class SessionChain;
  friend class SessionChainExecution;
  SessionLoop(InferenceSession& session, SessionLoopOptions options,
              StatefulProviderBinding cache_provider, StatefulProviderBinding sampler_provider);
  Status ValidateInputs(const SessionLoopInputs& inputs) const;
  Status CreateExecutionWithCache(SessionLoopInputs inputs, std::shared_ptr<KVCacheManager> cache,
                                  std::shared_ptr<TokenGenerationState> generation,
                                  std::shared_ptr<StatefulDispatch> sampler,
                                  std::unique_ptr<SessionLoopExecution>& out) const;

  InferenceSession& session_;
  const SessionLoopOptions options_;
  InlinedVector<std::string> fetch_names_;
  const StatefulProviderBinding cache_provider_;
  const StatefulProviderBinding sampler_provider_;
};

// Only Cancel may run concurrently with Step/Run. Reset and execution are sequential.
class SessionLoopExecution {
 public:
  Status Step(SessionLoopStepResult& out);
  Status Run(SessionLoopStepResult& out);
  void Cancel() noexcept { cancelled_.store(true); }
  Status Reset();
  bool RequiresReset() const { return requires_reset_; }
  const KVCacheManager* Cache() const { return cache_.get(); }
  const TokenGenerationState* Generation() const { return generation_.get(); }
  Status GetState(LoopFeeds& out) const;
  Status GetDispatchInfo(OrtStatefulProviderFeature feature, size_t component,
                         OrtStatefulProviderOperation operation, OrtStatefulDispatchInfo& out) const;
  Status GetStagingInfo(size_t component, OrtStatefulStagingInfo& out) const;
  const OrtStatefulStagingInfo& StagingInfo() const { return staging_; }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SessionLoopExecution);

 private:
  friend class SessionLoop;
  SessionLoopExecution(std::shared_ptr<const SessionLoop> loop, SessionLoopInputs inputs,
                       std::shared_ptr<KVCacheManager> cache, std::shared_ptr<TokenGenerationState> generation,
                       std::shared_ptr<StatefulDispatch> sampler,
                       bool owns_state);
  void EvaluateExits(const InlinedVector<OrtValue>& values);

  const std::shared_ptr<const SessionLoop> loop_;
  const SessionLoopInputs initial_inputs_;
  const std::shared_ptr<KVCacheManager> cache_;
  const std::shared_ptr<TokenGenerationState> generation_;
  const std::shared_ptr<StatefulDispatch> sampler_;
  const bool owns_state_;
  InlinedVector<OrtValue> feedback_;
  SessionLoopStepResult progress_;
  std::atomic<bool> cancelled_{false};
  bool requires_reset_ = false;
  OrtStatefulStagingInfo staging_{sizeof(OrtStatefulStagingInfo), 0, 0, 0};
};

}  // namespace onnxruntime
