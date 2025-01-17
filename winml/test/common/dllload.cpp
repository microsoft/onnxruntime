// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "Std.h"
#include "fileHelpers.h"
#include <winstring.h>
#include "dllload.h"

extern "C" {
HRESULT __stdcall OS_RoGetActivationFactory(HSTRING classId, GUID const& iid, void** factory) noexcept;
}

#ifdef _M_IX86
#pragma comment(linker, "/alternatename:_OS_RoGetActivationFactory@12=_RoGetActivationFactory@12")
#else
#pragma comment(linker, "/alternatename:OS_RoGetActivationFactory=RoGetActivationFactory")
#endif

bool starts_with(std::wstring_view value, std::wstring_view match) noexcept {
  return 0 == value.compare(0, match.size(), match);
}

HRESULT __stdcall WINRT_RoGetActivationFactory(HSTRING classId_hstring, GUID const& iid, void** factory) noexcept {
  *factory = nullptr;
  std::wstring_view name{WindowsGetStringRawBuffer(classId_hstring, nullptr), WindowsGetStringLen(classId_hstring)};
  HMODULE library{nullptr};

  std::wostringstream dll;
  dll << BINARY_NAME;

  std::wstring winml_dll_name = dll.str();
  std::wstring winml_dll_path = FileHelpers::GetWinMLPath() + winml_dll_name;
  std::wstring winml_dll_prefix = winml_dll_name.substr(0, winml_dll_name.size() - 3);
  if (starts_with(name, winml_dll_prefix)) {
    const wchar_t* lib_path = winml_dll_path.c_str();
#if WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP)
    library = LoadLibraryExW(lib_path, nullptr, 0);
#else
    library = LoadPackagedLibrary(lib_path, 0);
#endif
  } else {
    return OS_RoGetActivationFactory(classId_hstring, iid, factory);
  }

  if (!library) {
    return HRESULT_FROM_WIN32(GetLastError());
  }

  using DllGetActivationFactory = HRESULT __stdcall(HSTRING classId, void** factory);
  auto call = reinterpret_cast<DllGetActivationFactory*>(GetProcAddress(library, "DllGetActivationFactory"));

  if (!call) {
    HRESULT const hr = HRESULT_FROM_WIN32(GetLastError());
    WINRT_VERIFY(FreeLibrary(library));
    return hr;
  }

  winrt::com_ptr<wf::IActivationFactory> activation_factory;
  HRESULT const hr = call(classId_hstring, activation_factory.put_void());

  if (FAILED(hr)) {
    WINRT_VERIFY(FreeLibrary(library));
    return hr;
  }

  if (winrt::guid(iid) != winrt::guid_of<wf::IActivationFactory>()) {
    return activation_factory->QueryInterface(iid, factory);
  }

  *factory = activation_factory.detach();
  return S_OK;
}

int32_t __stdcall WINRT_RoGetActivationFactory(void* classId, winrt::guid const& iid, void** factory) noexcept {
  return WINRT_RoGetActivationFactory((HSTRING)classId, (GUID)iid, factory);
}
