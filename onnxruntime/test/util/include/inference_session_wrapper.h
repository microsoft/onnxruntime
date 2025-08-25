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
  // Expose the constructors from InferenceSession
  using InferenceSession::InferenceSession;

  const Graph& GetGraph() const {
    return model_->MainGraph();
  }

  Graph& GetMutableGraph() const {
    return model_->MainGraph();
  }

  const SessionState& GetSessionState() const {
    return InferenceSession::GetSessionState();
  }

  const Model& GetModel() const {
    return *model_;
  }
};

}  // namespace test
}  // namespace onnxruntime
