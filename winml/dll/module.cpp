// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "pch.h"
#include <windows.h>
#include <Hstring.h>

#include "LearningModelDevice.h"
#include "OnnxruntimeProvider.h"
#include "LearningModelBuilder.h"
#include "LearningModelOperatorResolutionPolicy.h"
#include "LearningModelOperator.h"

using namespace winrt::Windows::AI::MachineLearning::implementation;

void __stdcall OnErrorReported(bool alreadyReported, wil::FailureInfo const& failure) WI_NOEXCEPT {
  if (!alreadyReported) {
    winrt::hstring message(failure.pszMessage ? failure.pszMessage : L"");
    telemetry_helper.LogRuntimeError(
        failure.hr,
        winrt::to_string(message),
        failure.pszFile,
        failure.pszFunction,
        failure.uLineNumber);
  }
}

extern "C" BOOL WINAPI DllMain(_In_ HINSTANCE hInstance, DWORD dwReason, _In_ void* lpvReserved) {
  switch (dwReason) {
    case DLL_PROCESS_ATTACH:
      DisableThreadLibraryCalls(hInstance);

      // Register the TraceLogging provider feeding telemetry.  It's OK if this fails;
      // trace logging calls just become no-ops.
      telemetry_helper.Register();
      wil::SetResultTelemetryFallback(&OnErrorReported);
      break;
    case DLL_PROCESS_DETACH:
      telemetry_helper.LogWinMLShutDown();
      // Unregister Trace Logging Provider feeding telemetry
      telemetry_helper.UnRegister();

#ifdef NDEBUG
      bool dynamicUnload = (lpvReserved == nullptr);

      //
      // The OS can reclaim memory more quickly and correctly during process shutdown.
      // Continue to do this on debug builds due to leak detection tracing.
      //
      if (dynamicUnload)
#endif
      {
        LearningModelDevice::DllUnload();
      }

      break;
  }

  return true;
}

extern "C" HRESULT WINAPI MLCreateOperatorRegistry(_COM_Outptr_ IMLOperatorRegistry** registry) try {
  winrt::com_ptr<WinML::IEngineFactory> engine_factory;
  WINML_THROW_IF_FAILED(CreateOnnxruntimeEngineFactory(engine_factory.put()));
  WINML_THROW_IF_FAILED(engine_factory->CreateCustomRegistry(registry));
  return S_OK;
}
CATCH_RETURN();

STDAPI DllCanUnloadNow() {
  // The windows.ai.machinelearning.dll should not be freed by
  // CoFreeUnusedLibraries since there can be outstanding COM object
  // references to many objects (AbiCustomRegistry, IMLOperatorKernelContext,
  // IMLOperatorTensor, etc) that are not reference counted in this path.
  //
  // In order to implement DllCanUnloadNow we would need to reference count
  // all of the instances of non-WinRT COM objects that have been shared
  // across the dll boundary or harden the boundary APIs to make sure to
  // additional outstanding references are not cached by callers.
  //
  // Identifying and curating the complete list of IUnknown based COM objects
  // that are shared out as a consequence of the MLCreateOperatorRegistry API
  // will be a complex task to complete in RS5.
  //
  // As a temporary workaround we simply prevent the windows.ai.machinelearning.dll
  // from unloading.
  //
  // There are no known code paths that rely on opportunistic dll unload.
  return S_FALSE;
}

int32_t WINRT_CALL WinmlMoreGetActivationFactory(void* classId, void** factory) noexcept {
  try {
    *factory = nullptr;
    uint32_t length{};
    wchar_t const* const buffer = WINRT_WindowsGetStringRawBuffer(classId, &length);
    std::wstring_view const name{buffer, length};

    auto requal = [](std::wstring_view const& left, std::wstring_view const& right) noexcept {
      return std::equal(left.rbegin(), left.rend(), right.rbegin(), right.rend());
    };

    if (requal(name, L"Windows.AI.MachineLearning.More.LearningModelBuilder")) {
      *factory = winrt::detach_abi(winrt::make<winrt::Windows::AI::MachineLearning::More::factory_implementation::LearningModelBuilder>());
      return 0;
    }

    if (requal(name, L"Windows.AI.MachineLearning.More.LearningModelOperator")) {
      *factory = winrt::detach_abi(winrt::make<winrt::Windows::AI::MachineLearning::More::factory_implementation::LearningModelOperator>());
      return 0;
    }

    if (requal(name, L"Windows.AI.MachineLearning.More.LearningModelOperatorResolutionPolicy")) {
      *factory = winrt::detach_abi(winrt::make<winrt::Windows::AI::MachineLearning::More::factory_implementation::LearningModelOperatorResolutionPolicy>());
      return 0;
    }

#ifdef _WRL_MODULE_H_
    return ::Microsoft::WRL::Module<::Microsoft::WRL::InProc>::GetModule().GetActivationFactory(static_cast<HSTRING>(classId), reinterpret_cast<::IActivationFactory**>(factory));
#else
    return winrt::hresult_class_not_available(name).to_abi();
#endif
  } catch (...) {
    return winrt::to_hresult();
  }
}

STDAPI DllGetActivationFactory(HSTRING classId, void** factory) {
  auto ret = WINRT_GetActivationFactory(classId, factory);

  if (ret != 0)
    return WinmlMoreGetActivationFactory(classId, factory);

  return 0;
}
