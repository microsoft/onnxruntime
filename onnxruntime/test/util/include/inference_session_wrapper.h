// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/session/inference_session.h"
#include "core/graph/model.h"

namespace onnxruntime {
namespace test {

// InferenceSession wrapper class for use in tests where we need access to the Graph and SessionState
class InferenceSessionWrapper : public InferenceSession {
 public:
  explicit InferenceSessionWrapper(const SessionOptions& session_options,
                                   const Environment& env) : InferenceSession(session_options, env) {
  }

  const Graph& GetGraph() const {
    return model_->MainGraph();
  }

  Graph& GetMutableGraph() const {
    return model_->MainGraph();
  }

  const SessionState& GetSessionState() const {
    return InferenceSession::GetSessionState();
  }
};

}  // namespace test
}  // namespace onnxruntime
