// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "LearningModelSession.g.h"

#include "LearningModelBinding.h"
#include "MLOperatorAuthor.h"
#include "WinML_Lock.h"
#include "core/providers/winml/winml_provider_factory.h"
#include "iengine.h"

namespace winrt::Windows::AI::MachineLearning::implementation {

struct LearningModelSession : LearningModelSessionT<LearningModelSession> {
  /* LearningModelSession constructors (MachineLearningContract 1). */
  LearningModelSession() = delete;

  LearningModelSession(
      winml::LearningModel const& model);

  LearningModelSession(
      winml::LearningModel const& model,
      winml::LearningModelDevice const& deviceToRunOn);

  /* LearningModelSession constructors (MachineLearningContract 2). */
  LearningModelSession(
      winml::LearningModel const& model,
      winml::LearningModelDevice const& deviceToRunOn,
      winml::LearningModelSessionOptions const& sessionOptions);

  /* IClosable methods. */
  void
  Close();

  /* LearningModelSession properties (MachineLearningContract 1). */
  wfc::IPropertySet
  EvaluationProperties();

  winml::LearningModel
  Model();

  winml::LearningModelDevice
  Device();

  /* LearningModelSession methods (MachineLearningContract 1). */
  winml::LearningModelEvaluationResult
  Evaluate(
      winml::LearningModelBinding binding,
      hstring const& correlationId);

  wf::IAsyncOperation<winml::LearningModelEvaluationResult>
  EvaluateAsync(
      winml::LearningModelBinding binding,
      hstring const correlationId);

  winml::LearningModelEvaluationResult
  EvaluateFeatures(
      wfc::IMap<hstring, wf::IInspectable> const features,
      hstring const correlationId);

  wf::IAsyncOperation<winml::LearningModelEvaluationResult>
  EvaluateFeaturesAsync(
      wfc::IMap<hstring, wf::IInspectable> const features,
      hstring const correlationId);

 public:
  /* Non-ABI methods */

  WinML::IEngine*
  GetEngine();

  void
  CheckClosed();

 private:
  void
  Initialize();

  WinML::IModel*
  GetOptimizedModel();

  WinML::IModel*
  GetOptimizedModel(bool should_close_model);

  uint64_t
  Run(
      winrt::com_ptr<winmlp::LearningModelBinding> bindingImpl);

  winml::LearningModelEvaluationResult
  GetResults(
      winrt::com_ptr<winmlp::LearningModelBinding> bindingImpl,
      hstring const& correlationId,
      uint64_t fenceValueForDML);

  void
  ApplyEvaluationProperties();

  void
  ToggleProfiler();

 private:
  com_ptr<WinML::IEngineFactory> engine_factory_;
  com_ptr<WinML::IEngine> engine_;

  using MLOperatorRegistry = std::unique_ptr<IMLOperatorRegistry, void (*)(IMLOperatorRegistry*)>;
  MLOperatorRegistry operator_registry_;

  winml::LearningModel model_;
  winml::LearningModelDevice device_;
  winml::LearningModelSessionOptions session_options_;
  wfc::IPropertySet evaluation_properties_;

  // Synchronization
  CWinMLLock session_creation_lock_;
  CWinMLLock evaluate_lock_;

  // is_first_evaluate_ is used as a heuristic to determine
  // when the dml upload heap can be trimmed.
  bool is_first_evaluate_ = true;
};

}  // namespace winrt::Windows::AI::MachineLearning::implementation

namespace winrt::Windows::AI::MachineLearning::factory_implementation {

struct LearningModelSession : LearningModelSessionT<LearningModelSession, implementation::LearningModelSession> {
};

}  // namespace winrt::Windows::AI::MachineLearning::factory_implementation
