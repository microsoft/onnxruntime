// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/windows/windowsml_interop.h"

#ifdef _WIN32

#include <Windows.h>
#include <mutex>
#include <string>

#include "core/common/logging/logging.h"

namespace onnxruntime {
namespace windowsml {

namespace {

// Function pointer types for WindowsML exports
using GetEpCatalogApiFn = const OrtEpCatalogApi*(ORT_API_CALL*)();
using GetVersionStringFn = const char*(ORT_API_CALL*)();

// Singleton state for WindowsML plugin
struct WindowsMLState {
  std::once_flag init_flag;
  std::once_flag shutdown_flag;
  HMODULE module_handle{nullptr};
  const OrtEpCatalogApi* catalog_api{nullptr};
  GetVersionStringFn get_version_fn{nullptr};
  bool is_available{false};
  bool is_shutdown{false};

  void Initialize() {
    std::call_once(init_flag, [this]() {
      LoadPlugin();
    });
  }

  void DoShutdown() {
    std::call_once(shutdown_flag, [this]() {
      is_shutdown = true;
      catalog_api = nullptr;
      get_version_fn = nullptr;
      is_available = false;

      if (module_handle != nullptr) {
        // Note: We intentionally don't FreeLibrary here because:
        // 1. The plugin may have registered EPs that are still in use
        // 2. Windows will clean up on process exit
        // 3. Unloading during shutdown can cause crashes if there are
        //    outstanding references to plugin objects
        module_handle = nullptr;
      }
    });
  }

 private:
  void LoadPlugin() {
    // Get the path to the current module (onnxruntime.dll)
    HMODULE ort_module = nullptr;
    if (!GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&WindowsMLState::LoadPlugin),
            &ort_module)) {
      LOGS_DEFAULT(INFO) << "WindowsML: Failed to get ORT module handle. Error: " << GetLastError();
      return;
    }

    // Use dynamic buffer allocation to avoid MAX_PATH limitations
    // Start with a reasonable size and grow if needed
    std::wstring module_path;
    DWORD buffer_size = 512;
    DWORD path_len = 0;

    do {
      module_path.resize(buffer_size);
      path_len = GetModuleFileNameW(ort_module, module_path.data(), buffer_size);
      if (path_len == 0) {
        LOGS_DEFAULT(INFO) << "WindowsML: Failed to get ORT module path. Error: " << GetLastError();
        return;
      }
      // If the buffer was too small, GetModuleFileNameW returns buffer_size and
      // GetLastError() returns ERROR_INSUFFICIENT_BUFFER
      if (path_len < buffer_size) {
        module_path.resize(path_len);
        break;
      }
      buffer_size *= 2;
    } while (buffer_size <= 32768);  // Reasonable upper limit

    if (path_len >= buffer_size) {
      LOGS_DEFAULT(INFO) << "WindowsML: Module path exceeds maximum supported length.";
      return;
    }

    // Find the directory containing onnxruntime.dll
    size_t last_slash_pos = module_path.rfind(L'\\');
    if (last_slash_pos == std::wstring::npos) {
      LOGS_DEFAULT(INFO) << "WindowsML: Invalid ORT module path.";
      return;
    }
    module_path.resize(last_slash_pos);

    // Try to load Microsoft.Windows.AI.MachineLearning.dll from the same directory
    // This DLL is part of the Windows App SDK ML component and exports OrtGetEpCatalogApi
    std::wstring plugin_path = module_path + L"\\Microsoft.Windows.AI.MachineLearning.dll";

    module_handle = LoadLibraryExW(plugin_path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (module_handle == nullptr) {
      // DLL not present - this is expected for non-Windows ML scenarios
      LOGS_DEFAULT(INFO) << "WindowsML: Microsoft.Windows.AI.MachineLearning.dll not found at: " << std::string(plugin_path.begin(), plugin_path.end());
      return;
    }

    LOGS_DEFAULT(INFO) << "WindowsML: Loaded Microsoft.Windows.AI.MachineLearning.dll successfully.";

    // Get the EP Catalog API
    auto get_catalog_api_fn = reinterpret_cast<GetEpCatalogApiFn>(
        GetProcAddress(module_handle, "OrtGetEpCatalogApi"));
    if (get_catalog_api_fn != nullptr) {
      catalog_api = get_catalog_api_fn();
      if (catalog_api != nullptr) {
        LOGS_DEFAULT(INFO) << "WindowsML: EP Catalog API initialized.";
      }
    }

    // Get the version string function (optional)
    get_version_fn = reinterpret_cast<GetVersionStringFn>(
        GetProcAddress(module_handle, "OrtGetWindowsMLVersion"));

    is_available = (catalog_api != nullptr);
  }
};

WindowsMLState& GetState() {
  static WindowsMLState state;
  return state;
}

}  // namespace

const OrtEpCatalogApi* GetEpCatalogApi() {
  auto& state = GetState();
  if (state.is_shutdown) {
    return nullptr;
  }
  state.Initialize();
  return state.catalog_api;
}

const char* GetVersionString() {
  auto& state = GetState();
  if (state.is_shutdown) {
    return nullptr;
  }
  state.Initialize();
  if (state.get_version_fn != nullptr) {
    return state.get_version_fn();
  }
  return nullptr;
}

bool IsAvailable() {
  auto& state = GetState();
  if (state.is_shutdown) {
    return false;
  }
  state.Initialize();
  return state.is_available;
}

void Shutdown() {
  GetState().DoShutdown();
}

}  // namespace windowsml
}  // namespace onnxruntime

#else  // !_WIN32

// Non-Windows platforms: WindowsML is not available

namespace onnxruntime {
namespace windowsml {

const OrtEpCatalogApi* GetEpCatalogApi() {
  return nullptr;
}

const char* GetVersionString() {
  return nullptr;
}

bool IsAvailable() {
  return false;
}

void Shutdown() {
  // No-op on non-Windows platforms
}

}  // namespace windowsml
}  // namespace onnxruntime

#endif  // _WIN32
