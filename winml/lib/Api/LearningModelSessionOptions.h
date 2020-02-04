// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "LearningModelSessionOptions.g.h"

namespace winrt::Windows::AI::MachineLearning::implementation {

struct LearningModelSessionOptions : LearningModelSessionOptionsT<LearningModelSessionOptions> {
  LearningModelSessionOptions() = default;

  LearningModelSessionOptions(const LearningModelSessionOptions& options);

  uint32_t BatchSizeOverride();
  void BatchSizeOverride(uint32_t value);

  bool CloseModelOnSessionCreation();
  void CloseModelOnSessionCreation(bool value);

 private:
  // The batch size override property is used to inform the engine when the developer
  // wants to explicitly set the batch size of a model to a fixed batch size.
  //
  // 0     : dont override the model batch definitions
  // 1...n : override the model with the given batch size
  //
  // This value is a unsigned value, and users are not allowed to override models with a free batch size.
  // If the model supports free dimensional batch sizes, the caller should provide 0, to not override.
  //
  // The default value here is 1 so that models with free dimension batch sizes (which is very common)
  // can be optimized to fixed sizes.
  uint32_t batch_size_override_ = 1;

  // The close model on session creation property is used to inform the engine when the developer
  // no longer needs the learningmodelsession after session creation.
  // The engine can use the learning model during session creation to move resources rather than make copies.
  //
  // True     : Move resources in the LearningModel in to the LearningModelSession
  // False    : Copy resources in the LearningModel to the LearningModelSession
  //
  // The default value here is False so that models are not automatically closed on session creation.
  bool close_model_on_session_creation_ = false;
};

}  // namespace winrt::Windows::AI::MachineLearning::implementation

namespace winrt::Windows::AI::MachineLearning::factory_implementation {
struct LearningModelSessionOptions : LearningModelSessionOptionsT<LearningModelSessionOptions, implementation::LearningModelSessionOptions> {
};
}  // namespace winrt::Windows::AI::MachineLearning::factory_implementation
