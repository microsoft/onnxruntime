// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "LearningModelEvaluationResult.g.h"

namespace winrt::Windows::AI::MachineLearning::implementation {
struct LearningModelEvaluationResult : LearningModelEvaluationResultT<
                                           LearningModelEvaluationResult,
                                           ILearningModelEvaluationResultNative> {
  LearningModelEvaluationResult() = default;

  hstring CorrelationId();
  void CorrelationId(const hstring& correlationId);

  int32_t ErrorStatus();
  void ErrorStatus(int32_t errorStatus);

  bool Succeeded();
  void Succeeded(bool succeeded);

  Windows::Foundation::Collections::IMapView<hstring, Windows::Foundation::IInspectable> Outputs();
  void Outputs(Windows::Foundation::Collections::IMapView<hstring, Windows::Foundation::IInspectable> outputs);

  // ILearningModelEvaluationResultNative
  STDMETHOD(GetOutput)
  (
      const wchar_t* name,
      UINT32 cchName,
      IUnknown** result);

  HRESULT SetOutputs(std::unordered_map<std::string, Windows::Foundation::IInspectable>&& outputs);

 private:
  hstring m_correlationId;
  int32_t m_errorStatus = 0;
  bool m_succeeded = false;
  std::unordered_map<std::string, Windows::Foundation::IInspectable> m_outputs;
};
}  // namespace winrt::Windows::AI::MachineLearning::implementation
