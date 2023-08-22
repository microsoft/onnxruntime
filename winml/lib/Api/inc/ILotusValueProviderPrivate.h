// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "iengine.h"

// ILotusValueProviderPrivate exposes a private Lotus interface to the engine so that it can retrieve tensor
// resources stored in winrt structures.

namespace _winml {

class PoolObjectWrapper;

enum class BindingType {
  kInput,
  kOutput
};

struct BindingContext {
  BindingType type = BindingType::kInput;
  winml::LearningModelSession session = nullptr;
  winml::ILearningModelFeatureDescriptor descriptor = nullptr;
  wfc::IPropertySet properties = nullptr;
  std::shared_ptr<PoolObjectWrapper> converter;
};

struct __declspec(uuid("27e2f437-0112-4693-849e-e04323a620fb")) __declspec(novtable) ILotusValueProviderPrivate
  : IUnknown {
  virtual HRESULT __stdcall GetValue(BindingContext& binding_context, _winml::IValue** out) = 0;
  virtual HRESULT __stdcall IsPlaceholder(bool* is_placeholder) = 0;
  virtual HRESULT __stdcall UpdateSourceResourceData(BindingContext& binding_context, _winml::IValue* value) = 0;
  virtual HRESULT __stdcall AbiRepresentation(wf::IInspectable& abi_representation) = 0;
};
}  // namespace _winml
