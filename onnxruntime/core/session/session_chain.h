// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <variant>

#include "core/session/session_loop.h"

namespace onnxruntime {

class SessionChainExecution;

enum class ChainSourceKind { ExternalInput,
                             ExternalSequence,
                             ComponentOutput,
                             State };

struct ChainValueSource {
  ChainSourceKind kind;
  std::string name;
  size_t component_index = 0;
};

struct ChainInputBinding {
  std::string input_name;
  ChainValueSource source;
};

struct ChainStateBinding {
  std::string state_name;
  std::string output_name;
};

struct SessionChainComponent {
  std::string name;
  std::variant<InferenceSession*, std::shared_ptr<const SessionLoop>> execution;
  InlinedVector<ChainInputBinding> inputs;
  InlinedVector<ChainStateBinding> state_outputs;
  InlinedVector<LoopKVCacheBinding, 1> kv_caches;
  std::optional<StatefulProviderSelection> cache_provider;
  std::optional<StatefulProviderSelection> sampler_provider;
};

enum class ChainOutputKind { ModelOutput,
                             GeneratedTokenIds };

struct ChainOutputBinding {
  std::string name;
  size_t component_index;
  std::string output_name;
  ChainOutputKind kind = ChainOutputKind::ModelOutput;
};

struct SessionChainOptions {
  InlinedVector<SessionChainComponent, 1> components;
  InlinedVector<ChainOutputBinding> outputs;
};

struct SessionChainInputs {
  LoopFeeds inputs;
  InlinedHashMap<std::string, InlinedVector<OrtValue>> sequences;
  LoopFeeds initial_state;
};

struct SessionChainStepResult {
  bool executed = false;
  bool completed = false;
  int64_t invocations = 0;
  int64_t generated_tokens = 0;
  size_t components_completed = 0;
  std::string component_name;
  SessionLoopStepResult component_result;
  // Current invocation's snapshots; final named chain ports are emitted only on completion.
  LoopFeeds component_outputs;
  LoopFeeds outputs;
};

// Sessions remain caller-owned; component definitions are retained by the chain.
class SessionChain : public std::enable_shared_from_this<SessionChain> {
 public:
  static Status Create(SessionChainOptions options, std::shared_ptr<const SessionChain>& out);
  Status CreateExecution(SessionChainInputs inputs, std::unique_ptr<SessionChainExecution>& out) const;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SessionChain);

 private:
  friend class SessionChainExecution;
  explicit SessionChain(SessionChainOptions options);
  Status Initialize();
  Status ValidateInputs(const SessionChainInputs& inputs) const;

  const SessionChainOptions options_;
  InlinedVector<std::shared_ptr<const SessionLoop>> loops_;
  InlinedVector<InferenceSession*> sessions_;
  InlinedHashSet<std::string> external_names_;
  InlinedHashSet<std::string> sequence_names_;
  InlinedHashSet<std::string> initial_state_names_;
  InlinedVector<KVCacheSpec> cache_specs_;
  bool has_generation_ = false;
  std::optional<StatefulProviderBinding> cache_provider_;
};

// Step/Run/Reset are sequential. Only Cancel may be called concurrently.
class SessionChainExecution {
 public:
  Status Step(SessionChainStepResult& out);
  Status Run(SessionChainStepResult& out);
  void Cancel() noexcept { cancelled_.store(true); }
  Status Reset();
  bool RequiresReset() const { return requires_reset_; }
  const LoopFeeds& State() const { return state_; }
  const KVCacheManager* Cache() const { return cache_.get(); }
  const TokenGenerationState* Generation() const { return generation_.get(); }
  Status GetDispatchInfo(OrtStatefulProviderFeature feature, size_t component,
                         OrtStatefulProviderOperation operation, OrtStatefulDispatchInfo& out) const;
  Status GetStagingInfo(size_t component, OrtStatefulStagingInfo& out) const;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SessionChainExecution);

 private:
  friend class SessionChain;
  SessionChainExecution(std::shared_ptr<const SessionChain> chain, SessionChainInputs inputs,
                        std::shared_ptr<KVCacheManager> cache, std::shared_ptr<TokenGenerationState> generation);
  Status Resolve(const ChainValueSource& source, const OrtValue*& out) const;
  Status PrepareComponentInputs(SessionLoopInputs& inputs) const;
  Status StepComponent(SessionChainStepResult& result);

  const std::shared_ptr<const SessionChain> chain_;
  const SessionChainInputs initial_inputs_;
  const std::shared_ptr<KVCacheManager> cache_;
  const std::shared_ptr<TokenGenerationState> generation_;
  InlinedVector<std::shared_ptr<StatefulDispatch>> samplers_;
  InlinedVector<OrtStatefulStagingInfo> staging_;
  InlinedVector<LoopFeeds> component_outputs_;
  LoopFeeds state_;
  std::unique_ptr<SessionLoopExecution> active_;
  size_t component_index_ = 0;
  int64_t invocations_ = 0;
  std::atomic<bool> cancelled_{false};
  bool requires_reset_ = false;
};

}  // namespace onnxruntime
