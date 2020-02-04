// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "iengine.h"

// ILotusValueProviderPrivate exposes a private Lotus interface to the engine so that it can retrieve tensor
// resources stored in winrt structures.

namespace Windows::AI::MachineLearning {

class PoolObjectWrapper;

enum class BindingType { kInput,
                         kOutput };

struct BindingContext {
  BindingType type = BindingType::kInput;
  winrt::Windows::AI::MachineLearning::LearningModelSession session = nullptr;
  winrt::Windows::AI::MachineLearning::ILearningModelFeatureDescriptor descriptor = nullptr;
  winrt::Windows::Foundation::Collections::IPropertySet properties = nullptr;
  std::shared_ptr<PoolObjectWrapper> converter;
};

struct __declspec(uuid("27e2f437-0112-4693-849e-e04323a620fb")) __declspec(novtable) ILotusValueProviderPrivate : IUnknown {
  virtual HRESULT __stdcall GetValue(BindingContext& binding_context, WinML::IValue** out) = 0;
  virtual HRESULT __stdcall IsPlaceholder(bool* is_placeholder) = 0;
  virtual HRESULT __stdcall UpdateSourceResourceData(BindingContext& binding_context, WinML::IValue* value) = 0;
  virtual HRESULT __stdcall AbiRepresentation(winrt::Windows::Foundation::IInspectable& abi_representation) = 0;
};

}  // namespace Windows::AI::MachineLearning