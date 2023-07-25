// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "LearningModelEvaluationResult.g.h"

namespace WINMLP {
struct LearningModelEvaluationResult
  : LearningModelEvaluationResultT<LearningModelEvaluationResult, ILearningModelEvaluationResultNative> {
  LearningModelEvaluationResult() = default;

  hstring CorrelationId();
  void CorrelationId(const hstring& correlationId);

  int32_t ErrorStatus();
  void ErrorStatus(int32_t errorStatus);

  bool Succeeded();
  void Succeeded(bool succeeded);

  wfc::IMapView<hstring, wf::IInspectable> Outputs();
  void Outputs(wfc::IMapView<hstring, wf::IInspectable> outputs);

  // ILearningModelEvaluationResultNative
  STDMETHOD(GetOutput)
  (const wchar_t* name, UINT32 cchName, IUnknown** result);

  HRESULT SetOutputs(std::unordered_map<std::string, wf::IInspectable>&& outputs);

 private:
  hstring m_correlationId;
  int32_t m_errorStatus = 0;
  bool m_succeeded = false;
  std::unordered_map<std::string, wf::IInspectable> m_outputs;
};
}  // namespace WINMLP
