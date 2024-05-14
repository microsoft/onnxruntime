// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "dll/pch.h"
#include <windows.h>
#include <Hstring.h>
#include "LearningModelDevice.h"
#include "OnnxruntimeProvider.h"

#ifndef BUILD_INBOX

#include "LearningModelBuilder.h"
#include "LearningModelOperator.h"
#include "LearningModelSessionOptionsExperimental.h"
#include "LearningModelSessionExperimental.h"
#include "LearningModelExperimental.h"
#include "LearningModelJoinOptions.h"

#define STRINGIFY(x) #x
#define XSTRINGIFY(x) STRINGIFY(x)

#endif

using namespace winmlp;

extern "C" BOOL WINAPI DllMain(_In_ HINSTANCE hInstance, DWORD dwReason, _In_ void* lpvReserved) {
  switch (dwReason) {
    case DLL_PROCESS_ATTACH:
      DisableThreadLibraryCalls(hInstance);

      // Register the TraceLogging provider feeding telemetry.  It's OK if this fails;
      // trace logging calls just become no-ops.
      telemetry_helper.Register();
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
  winrt::com_ptr<_winml::IEngineFactory> engine_factory;
  WINML_THROW_IF_FAILED(CreateOnnxruntimeEngineFactory(engine_factory.put()));
  WINML_THROW_IF_FAILED(engine_factory->CreateCustomRegistry(registry));
  return S_OK;
}
CATCH_RETURN();

__control_entrypoint(DllExport) STDAPI DllCanUnloadNow() {
  // This dll should not be freed by
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
  // As a temporary workaround we simply prevent the dll from unloading.
  //
  // There are no known code paths that rely on opportunistic dll unload.
  return S_FALSE;
}

#ifndef BUILD_INBOX
STDAPI DllGetExperimentalActivationFactory(void* classId, void** factory) noexcept {
  try {
    *factory = nullptr;
    std::wstring_view const name{*reinterpret_cast<winrt::hstring*>(&classId)};

    auto requal = [](std::wstring_view const& left, std::wstring_view const& right) noexcept {
      return std::equal(left.rbegin(), left.rend(), right.rbegin(), right.rend());
    };

    std::wostringstream learning_model_builder_class;
    learning_model_builder_class
      << XSTRINGIFY(WINML_ROOT_NS) << ".AI.MachineLearning.Experimental.LearningModelBuilder";
    if (requal(name, learning_model_builder_class.str())) {
      *factory = winrt::detach_abi(winrt::make<WINML_EXPERIMENTAL::factory_implementation::LearningModelBuilder>());
      return 0;
    }

    std::wostringstream learning_model_operator_class;
    learning_model_operator_class
      << XSTRINGIFY(WINML_ROOT_NS) << ".AI.MachineLearning.Experimental.LearningModelOperator";
    if (requal(name, learning_model_operator_class.str())) {
      *factory = winrt::detach_abi(winrt::make<WINML_EXPERIMENTAL::factory_implementation::LearningModelOperator>());

      return 0;
    }

    std::wostringstream learning_model_session_experimental_class;
    learning_model_session_experimental_class
      << XSTRINGIFY(WINML_ROOT_NS) << ".AI.MachineLearning.Experimental.LearningModelSessionExperimental";
    if (requal(name, learning_model_session_experimental_class.str())) {
      *factory =
        winrt::detach_abi(winrt::make<WINML_EXPERIMENTAL::factory_implementation::LearningModelSessionExperimental>());
      return 0;
    }

    std::wostringstream learning_model_experimental_class;
    learning_model_experimental_class
      << XSTRINGIFY(WINML_ROOT_NS) << ".AI.MachineLearning.Experimental.LearningModelExperimental";
    if (requal(name, learning_model_experimental_class.str())) {
      *factory =
        winrt::detach_abi(winrt::make<WINML_EXPERIMENTAL::factory_implementation::LearningModelExperimental>());
      return 0;
    }

    std::wostringstream learning_model_join_options_class;
    learning_model_join_options_class
      << XSTRINGIFY(WINML_ROOT_NS) << ".AI.MachineLearning.Experimental.LearningModelJoinOptions";
    if (requal(name, learning_model_join_options_class.str())) {
      *factory = winrt::detach_abi(winrt::make<WINML_EXPERIMENTAL::factory_implementation::LearningModelJoinOptions>());
      return 0;
    }

    std::wostringstream learning_model_session_options_experimental_class;
    learning_model_session_options_experimental_class
      << XSTRINGIFY(WINML_ROOT_NS) << ".AI.MachineLearning.Experimental.LearningModelSessionOptionsExperimental";
    if (requal(name, learning_model_session_options_experimental_class.str())) {
      *factory = winrt::detach_abi(
        winrt::make<WINML_EXPERIMENTAL::factory_implementation::LearningModelSessionOptionsExperimental>()
      );
      return 0;
    }

    return winrt::hresult_class_not_available(name).to_abi();
  } catch (...) {
    return winrt::to_hresult();
  }
}
#endif

STDAPI DllGetActivationFactory(HSTRING classId, void** factory) {
  auto ret = WINRT_GetActivationFactory(classId, factory);

#ifndef BUILD_INBOX
  if (ret != 0) {
    return DllGetExperimentalActivationFactory(classId, factory);
  }
#endif

  return ret;
}

// LoadLibraryW isn't support on Windows 8.1. This is a workaround so that CppWinRT calls this function for loading libraries
void* __stdcall WINRT_IMPL_LoadLibraryW(wchar_t const* name) noexcept {
#if WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP)
  return LoadLibraryExW(name, nullptr, 0);
#else
  return LoadPackagedLibrary(name, 0);
#endif
}
