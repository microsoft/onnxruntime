#include "pch.h"
#include <windows.h>
#include <Hstring.h>
#include "WinMLProfiler.h"

#include "LearningModelDevice.h"
#include "AbiCustomRegistryImpl.h"
#include "core/framework/customregistry.h"

using namespace winrt::Windows::AI::MachineLearning::implementation;

#ifdef false
// TODO: Delete all defines in this ifndef block after changing to a newer version of
// WIL that is compatible with the trace logging functionality of the SDK packages.
// https://microsoft.visualstudio.com/OS/_workitems/edit/21313649
#ifndef WINDOWSAI_RAZZLE_BUILD
#define _TlgActivityDecl _tlgActivityDecl
#define _TlgActivityRef _tlgActivityRef
#define _TlgKeywordVal _tlgKeywordVal
#define _TlgActivity_Keyword _tlgActivity_Keyword
#define _TlgActivity_Level _tlgActivity_Level
#define _TlgActivityPrivacyTag _tlgActivityPrivacyTag
#define _TlgLevelVal _tlgLevelVal
#define _TLG_FOREACH _tlg_FOREACH
#define _TLG_CASSERT static_assert
#define _TlgDefineProvider_annotation(hProvider, functionPostfix, requiresWrapper, providerName) \
    _TlgDefineProvider_functionWrapperBegin##requiresWrapper(functionPostfix) \
        __annotation( \
            L"_TlgDefineProvider:|" _TLG_PASTE(L, _TLG_STRINGIZE(__LINE__)) L"|" _TLG_PASTE(L, _TLG_STRINGIZE(hProvider)) L"|" \
            L##providerName \
        ); \
    _TlgDefineProvider_functionWrapperEnd##requiresWrapper
#define _TlgDefineProvider_functionWrapperBegin0(functionPostfix)
#define _TlgDefineProvider_functionWrapperBegin1(functionPostfix) static void __cdecl _TLG_PASTE(_TlgDefineProvider_annotation_, functionPostfix)(void) \
                                                {
#define _TlgDefineProvider_functionWrapperEnd0
#define _TlgDefineProvider_functionWrapperEnd1  }
#endif
#include <wil/TraceLogging.h>


class WindowsMLWilProvider : public wil::TraceLoggingProvider
{
    IMPLEMENT_TRACELOGGING_CLASS(WindowsMLWilProvider,
        WINML_PROVIDER_DESC,
        WINML_PROVIDER_GUID);
    virtual void OnErrorReported(bool alreadyReported, wil::FailureInfo const &failure) WI_NOEXCEPT;

};

void WindowsMLWilProvider::OnErrorReported(bool alreadyReported, wil::FailureInfo const &failure) WI_NOEXCEPT
{
    if (!alreadyReported)
    {
        winrt::hstring message(failure.pszMessage? failure.pszMessage : L"");
        g_Telemetry.LogRuntimeError(failure.hr, winrt::to_string(message), failure.pszFile, failure.pszFunction, failure.uLineNumber);
    }
}

#endif 

extern "C" BOOL WINAPI DllMain(_In_ HINSTANCE hInstance, DWORD dwReason, _In_ void* lpvReserved)
{
    switch (dwReason)
    {
    case DLL_PROCESS_ATTACH:
        DisableThreadLibraryCalls(hInstance);

        // Register the TraceLogging provider feeding telemetry.  It's OK if this fails;
        // trace logging calls just become no-ops.
        g_Telemetry.Register();

        // Log Dll load
        g_Telemetry.LogDllAttachEvent();
        // Enable Profiling if the device is sampled at measure level
        if (g_Telemetry.IsMeasureSampled())
        {
            g_Profiler.Enable(ProfilerType::CPU);
            g_Profiler.Reset(ProfilerType::CPU);
        }
        // wil::SetResultTelemetryFallback(WindowsMLWilProvider::FallbackTelemetryCallback);
        break;
    case DLL_PROCESS_DETACH:
        g_Telemetry.LogRuntimePerf(g_Profiler, true);
        // Unregister Trace Logging Provider feeding telemetry
        g_Telemetry.UnRegister();

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

extern "C" HRESULT WINAPI MLCreateOperatorRegistry(_COM_Outptr_ IMLOperatorRegistry** registry) try
{
    *registry = nullptr;

    auto operatorRegistry = wil::MakeOrThrow<winrt::Windows::AI::MachineLearning::implementation::AbiCustomRegistryImpl>();

    *registry = operatorRegistry.Detach();
    return S_OK;
}
CATCH_RETURN();

STDAPI DllCanUnloadNow()
{
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

STDAPI DllGetActivationFactory(HSTRING classId, void** factory)
{
    return WINRT_GetActivationFactory(classId, factory);
}