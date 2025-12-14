// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "winappsdk_bootstrap.h"


#include <core/session/onnxruntime_c_api.h>
#include <core/session/onnxruntime_cxx_api.h>

// #include <vector>

#include <string>
#include <locale>
#include <codecvt>
#include <iostream>

#include <filesystem>
#include <iosfwd>
#include <random>

// #include <Unknwn.h>
#include <winrt/base.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Microsoft.Windows.AI.MachineLearning.h>

#include <appmodel.h>

#ifndef BUILD_WINAPPSDK_PERF_TEST
#error "This file should only be compiled when BUILD_WINAPPSDK_PERF_TEST is ON"
#endif

#if defined(USE_CUDA) || defined(USE_TENSORRT) || defined(USE_NV) || defined(USE_OPENVINO) || defined(USE_DML)
#error "None of this should be defined or compiled.
#endif

#ifdef ABSL_FLAGS_STRIP_NAMES
#if ABSL_FLAGS_STRIP_NAMES != 0
// ABSL_FLAGS_STRIP_NAMES is set to 1 by default to disable flag registration when building for Android, iPhone, and "embedded devices".
// See the issue: https://github.com/abseil/abseil-cpp/issues/1875
#error "ABSL_FLAGS_STRIP_NAMES should be 0, not 1"
#endif
#endif

using namespace winrt::Windows::Foundation::Collections;

static wchar_t* g_packageFullName = nullptr;
static wchar_t* g_packageDependencyId = nullptr;
static HRESULT g_initializationResult = E_NOT_VALID_STATE;
static PACKAGEDEPENDENCY_CONTEXT g_packageContext = nullptr;

void WinAppSDK_FindAndRegisterAllProviders();

static inline bool IsRunningOnArm64()
{
#if defined(_M_ARM64EC) || defined(_M_ARM64)
    return true;
#else
    static const bool isArm64Native = [] {
        USHORT processMachine{};
        USHORT nativeMachine{};
        const auto result{::IsWow64Process2(GetCurrentProcess(), &processMachine, &nativeMachine)};
        return (0 == result) || (nativeMachine == IMAGE_FILE_MACHINE_ARM64);
    }();
    return isArm64Native;
#endif
}

static inline PackageDependencyProcessorArchitectures GetPackageDependencyProcessorArchitectures()
{
#if defined(_M_ARM64)
    const PackageDependencyProcessorArchitectures architectures = PackageDependencyProcessorArchitectures_Arm64;
#elif defined(_M_X64)
    const PackageDependencyProcessorArchitectures architectures = PackageDependencyProcessorArchitectures_X64;
#elif defined(_M_ARM64EC)
    const PackageDependencyProcessorArchitectures architectures =
        IsRunningOnArm64() ? PackageDependencyProcessorArchitectures_Arm64 : PackageDependencyProcessorArchitectures_X64;
#endif
    return architectures;
}

static std::wstring g_expectedFrameworkFamilyName;

HRESULT WinAppSDK_WinMLInitialize(const char* const winappsdk_version)
{
    if (g_packageDependencyId != nullptr)
    {
        return g_initializationResult;
    }

    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> to_wstring_converter;

    std::wstring packageFamilyName = L"Microsoft.WindowsAppRuntime." +
        to_wstring_converter.from_bytes(winappsdk_version)
        + L"_8wekyb3d8bbwe";

    std::wcout << "[WinAppSDK] package_family_name:" << packageFamilyName << std::endl;

    // Create the package dependency
    {
        PSID userContext = nullptr;

        PACKAGE_VERSION minVersion {};

        const PackageDependencyProcessorArchitectures architectures = GetPackageDependencyProcessorArchitectures();

        const PackageDependencyLifetimeKind lifetimeKind = PackageDependencyLifetimeKind_Process;

        const wchar_t* lifetimeArtifact = nullptr;

        CreatePackageDependencyOptions options = CreatePackageDependencyOptions_None;

        HRESULT result = S_OK;
        result = TryCreatePackageDependency(
            userContext,
            packageFamilyName.c_str(),
            minVersion,
            architectures,
            lifetimeKind,
            lifetimeArtifact,
            options,
            &g_packageDependencyId);

        if (FAILED(result))
        {
            return result;
        }

        if (!g_packageDependencyId)
        {
            return E_UNEXPECTED;
        }
    }

    // Add the package dependency
    {
        int rank = 0;
        AddPackageDependencyOptions options = AddPackageDependencyOptions_PrependIfRankCollision;

        g_initializationResult = AddPackageDependency(g_packageDependencyId, rank, options, &g_packageContext, &g_packageFullName);
        g_expectedFrameworkFamilyName = packageFamilyName;
    }

    return g_initializationResult;
}

void WinAppSDK_WinMLInitializeMLAndRegisterAllProviders(const char* const winappsdk_version)
{
    HRESULT hr = WinAppSDK_WinMLInitialize(winappsdk_version);
    if (FAILED(hr))
    {
        throw winrt::hresult_error(hr);
    }

    WinAppSDK_FindAndRegisterAllProviders();
}

void WinAppSDK_WinMLUninitialize()
{
    if (g_packageDependencyId)
    {
        ::HeapFree(::GetProcessHeap(), 0, g_packageDependencyId);
        g_packageDependencyId = nullptr;
    }

    if (g_packageFullName)
    {
        ::HeapFree(::GetProcessHeap(), 0, g_packageFullName);
        g_packageFullName = nullptr;
    }

    if (g_packageContext)
    {
        ::RemovePackageDependency(g_packageContext);
        g_packageContext = nullptr;
    }

    g_initializationResult = E_NOT_VALID_STATE;
}

extern "C" const OrtApiBase* __cdecl OrtGetApiBase() noexcept
{
    // Static local variable with guaranteed one-time initialization (C++11)
    static const OrtApiBase* s_ortApiBase = []() -> const OrtApiBase* {

        if (g_expectedFrameworkFamilyName.empty())
        {
            return nullptr;
        }

        // Find framework package in current process dependency graph
        UINT32 bufferLength = 0;
        LONG rc = GetCurrentPackageInfo(PACKAGE_FILTER_DYNAMIC | PACKAGE_FILTER_STATIC, &bufferLength, nullptr, nullptr);
        if (rc != ERROR_INSUFFICIENT_BUFFER)
        {
            return nullptr;
        }

        auto buffer = std::make_unique<BYTE[]>(bufferLength);
        UINT32 count = 0;
        rc = GetCurrentPackageInfo(PACKAGE_FILTER_DYNAMIC | PACKAGE_FILTER_STATIC, &bufferLength, buffer.get(), &count);
        if (rc != ERROR_SUCCESS)
        {
            return nullptr;
        }

        // Look for framework package in dependency graph
        auto packageInfos = reinterpret_cast<PACKAGE_INFO*>(buffer.get());
        const WCHAR* frameworkPackagePath = nullptr;

        for (UINT32 i = 0; i < count; ++i)
        {
            const PACKAGE_INFO& packageInfo = packageInfos[i];
            if (packageInfo.packageFamilyName && packageInfo.path &&
                _wcsicmp(packageInfo.packageFamilyName, g_expectedFrameworkFamilyName.c_str()) == 0)
            {
                frameworkPackagePath = packageInfo.path;
                break;
            }
        }

        if (!frameworkPackagePath)
        {
            return nullptr;
        }

        // Build path to onnxruntime.dll and load it
        std::wstring onnxRuntimePath = std::wstring(frameworkPackagePath) + L"\\onnxruntime.dll";
        HMODULE onnxruntimeModule = LoadLibraryExW(onnxRuntimePath.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (!onnxruntimeModule)
        {
            return nullptr;
        }

        // Get the actual path of the loaded module
        wchar_t actualPath[MAX_PATH];
        DWORD pathLength = GetModuleFileNameW(onnxruntimeModule, actualPath, MAX_PATH);
        if (pathLength > 0 && pathLength < MAX_PATH)
        {
            std::wcout << L"[WinAppSDK:OrtGetApiBase] " << actualPath << std::endl;
        }
        else
        {
            std::wcout << L"Failed to get module path for onnxruntimeModule" << std::endl;
        }

        // Get the real OrtGetApiBase function and call it
        using OrtGetApiBaseFunc = OrtApiBase* (*)();
        auto ortGetApiBase = reinterpret_cast<OrtGetApiBaseFunc>(GetProcAddress(onnxruntimeModule, "OrtGetApiBase"));
        if (ortGetApiBase)
        {
            return ortGetApiBase();
        }

        return nullptr;
    }();

    return s_ortApiBase;
}

const char* ensure_ready_result_string[] = {
    "InProgress",
    "Success",
    "Failure"};

const char* execution_provider_ready_state_string[] = {
    "Ready",
    "NotReady",
    "NotPresent"};

void WinAppSDK_FindAndRegisterAllProviders()
{
    auto catalog = winrt::Microsoft::Windows::AI::MachineLearning::ExecutionProviderCatalog::GetDefault();

    auto providers = catalog.FindAllProviders();
    for (const auto& provider : providers) {
        std::wcout << "[WinAppSDK] Provider: " << provider.Name().c_str();
        auto readyState = provider.ReadyState();
        std::wcout << "[WinAppSDK]  Ready state: " << execution_provider_ready_state_string[static_cast<int>(readyState)] << std::endl;

        auto ensure_ready_result = provider.EnsureReadyAsync().get();
        std::wcout << "[WinAppSDK]  ensure_ready_result:" << ensure_ready_result_string[static_cast<int>(ensure_ready_result.Status())];
        std::wcout << " DiagnosticText:" << ensure_ready_result.DiagnosticText().c_str() << std::endl;

        auto registration_result = provider.TryRegister();
        std::wcout << "[WinAppSDK]  registration_result:" << (registration_result ? "true" : "false") << std::endl;
    }
}
