// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <Windows.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cwchar>
#include <memory>
#include <string>
#include <string_view>

namespace {

struct CloseWin32Handle {
  void operator()(HANDLE handle) const { CloseHandle(handle); }
};
using UniqueHandle = std::unique_ptr<void, CloseWin32Handle>;

int Fail(const char* message) {
  std::fprintf(stderr, "%s (Win32 error: %lu)\n", message, GetLastError());
  return EXIT_FAILURE;
}

int RunChild(const wchar_t* library_path, std::wstring_view mode, HANDLE verified) {
  if (!SetEnvironmentVariableW(L"ORT_DISABLE_TELEMETRY", L"1")) {
    return Fail("Could not disable network telemetry");
  }
  const HMODULE library = LoadLibraryW(library_path);
  if (!library) {
    return Fail("LoadLibraryW failed");
  }
  using RunTest = void (*)(bool, bool, HANDLE);
  const auto run_test = reinterpret_cast<RunTest>(GetProcAddress(library, "OrtTestTelemetryShutdown"));
  if (!run_test) {
    return Fail("GetProcAddress failed");
  }
  const bool process_exit = mode == L"process_exit";
  run_test(process_exit, mode == L"normal_destruction", verified);
  const DWORD expected_signal = mode == L"normal_destruction" ? WAIT_OBJECT_0 : WAIT_TIMEOUT;
  if (WaitForSingleObject(verified, 0) != expected_signal) {
    return Fail("Telemetry destruction happened at the wrong lifecycle stage");
  }
  if (process_exit) {
    ExitProcess(EXIT_SUCCESS);
  }
  if (!FreeLibrary(library)) {
    return Fail("FreeLibrary failed");
  }
  return EXIT_SUCCESS;
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
  if (argc == 5 && std::wstring_view(argv[1]) == L"--child") {
    const auto verified = reinterpret_cast<HANDLE>(static_cast<uintptr_t>(std::wcstoull(argv[4], nullptr, 10)));
    return RunChild(argv[2], argv[3], verified);
  }
  if (argc != 3) {
    return Fail("Expected test DLL path and lifecycle mode");
  }
  const std::wstring_view mode(argv[2]);
  if (mode != L"process_exit" && mode != L"normal_destruction" && mode != L"explicit_unload") {
    return Fail("Unknown lifecycle mode");
  }

  // The parent checks a signal from the DLL destructor, so an early child exit
  // cannot pass without actually exercising and verifying telemetry teardown.
  SECURITY_ATTRIBUTES attributes{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
  UniqueHandle verified(CreateEventW(&attributes, TRUE, FALSE, nullptr));
  if (!verified) {
    return Fail("CreateEventW failed");
  }
  std::wstring executable(32768, L'\0');
  const DWORD length = GetModuleFileNameW(nullptr, executable.data(), static_cast<DWORD>(executable.size()));
  if (length == 0 || length >= executable.size()) {
    return Fail("GetModuleFileNameW failed");
  }
  executable.resize(length);
  std::wstring command = L"\"" + executable + L"\" --child \"" + argv[1] + L"\" " + argv[2] + L" " +
                         std::to_wstring(reinterpret_cast<uintptr_t>(verified.get()));
  STARTUPINFOW startup{};
  startup.cb = sizeof(startup);
  PROCESS_INFORMATION info{};
  if (!CreateProcessW(executable.c_str(), command.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW,
                      nullptr, nullptr, &startup, &info)) {
    return Fail("CreateProcessW failed");
  }
  UniqueHandle process(info.hProcess);
  UniqueHandle thread(info.hThread);
  if (WaitForSingleObject(process.get(), 10000) != WAIT_OBJECT_0) {
    TerminateProcess(process.get(), EXIT_FAILURE);
    WaitForSingleObject(process.get(), 2000);
    return Fail("Child process did not exit within 10 seconds");
  }
  DWORD exit_code = EXIT_FAILURE;
  if (!GetExitCodeProcess(process.get(), &exit_code)) {
    return Fail("GetExitCodeProcess failed");
  }
  if (exit_code != EXIT_SUCCESS) {
    std::fprintf(stderr, "Child process failed with exit code %lu\n", exit_code);
    return EXIT_FAILURE;
  }
  if (WaitForSingleObject(verified.get(), 0) != WAIT_OBJECT_0) {
    return Fail("Child did not verify telemetry destruction");
  }
  return EXIT_SUCCESS;
}
