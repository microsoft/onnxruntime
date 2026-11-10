// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <optional>

#include "core/common/inlined_containers.h"
#include "core/framework/ort_value.h"

namespace onnxruntime {

class NodeArg;

class Sampler {
 public:
  // CPU-only; any NaN is an error. Equal maxima select the lowest token ID.
  static Status Greedy(const OrtValue& logits, std::optional<int64_t> position, int64_t& token);
};

enum class TokenGenerationPhase { Prefill,
                                  Decode,
                                  Unspecified };

struct TokenGenerationOptions {
  TokenGenerationPhase phase = TokenGenerationPhase::Unspecified;
  std::string input_ids;
  std::string logits_output;
  std::string attention_mask;
  std::string position_ids;
  std::optional<int64_t> logits_position;
};

struct TokenGenerationState {
  InlinedVector<int64_t> token_ids;
  int64_t processed_tokens = 0;
  bool prefill_complete = false;
  void Reset() {
    token_ids.clear();
    processed_tokens = 0;
    prefill_complete = false;
  }
};

class TokenGeneration {
 public:
  static Status Validate(const TokenGenerationOptions& options,
                         gsl::span<const NodeArg* const> inputs, gsl::span<const NodeArg* const> outputs);
  static InlinedVector<std::string> ManagedInputs(const TokenGenerationOptions& options);
  static Status ValidatePrompt(const OrtValue& prompt);
  static Status PrepareInputs(const TokenGenerationOptions& options, const TokenGenerationState& state,
                              const OrtValue* prompt, AllocatorPtr allocator,
                              InlinedHashMap<std::string, OrtValue>& inputs, int64_t& input_count);
  static OrtValue OutputTokens(const TokenGenerationState& state, AllocatorPtr allocator);
};

}  // namespace onnxruntime
