// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include <Windows.h>
#include <stdlib.h>
#include <filesystem>
#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_cxx_api.h"

// This program is to test the delay loading of onnxruntime.dll.
//
// To verify the delay loading actually works, we need to do the test in 2 steps:
//
// 1. Prepare a folder structure like below:
//
//    ├── webgpu_delay_load_test_root (newly created folder)
//    │   ├── dlls
//    │   │   ├── onnxruntime.dll
//    │   │   ├── webgpu_dawn.dll
//    │   │   ├── dxil.dll
//    │   │   └── dxcompiler.dll
//    │   └── test.exe
//    └── onnxruntime_webgpu_delay_load_test.exe (this binary)
//
//    This folder structure ensures no DLLs are in the same folder as the executable (test.exe).
//
// 2. Launch the test binary from the root folder of the above structure.
//
// So, there are 2 modes of this program:
// 1. "Prepare" mode: Do the step 1 above. (default)
// 2. "Test" mode: Do the step 2 above. (specified by --test argument)

int prepare_main();
int test_main();

int wmain(int argc, wchar_t* argv[]) {
  if (argc == 2 && wcscmp(argv[1], L"--test") == 0) {
    return test_main();
  } else {
    return prepare_main();
  }
}

int prepare_main() {
  std::wstring path_str(32768, L'\0');
  GetModuleFileNameW(NULL, path_str.data(), static_cast<DWORD>(path_str.size()));

  namespace fs = std::filesystem;
  fs::path exe_full_path{path_str};                                    // <TEST_DIR>/onnxruntime_webgpu_delay_load_test.exe
  fs::path test_dir = exe_full_path.parent_path();                     // <TEST_DIR>/
  fs::path exe_name = exe_full_path.filename();                        // onnxruntime_webgpu_delay_load_test.exe
  fs::path root_folder = test_dir / L"webgpu_delay_load_test_root\\";  // <TEST_DIR>/webgpu_delay_load_test_root/
  fs::path dlls_folder = root_folder / L"dlls\\";                      // <TEST_DIR>/webgpu_delay_load_test_root/dlls/

  // ensure the test folder exists and is empty
  if (fs::exists(root_folder)) {
    fs::remove_all(root_folder);
  }
  fs::create_directories(dlls_folder);

  fs::current_path(test_dir);

  // copy the required DLLs to the dlls folder
  fs::copy_file(L"onnxruntime.dll", dlls_folder / L"onnxruntime.dll");
  fs::copy_file(L"dxil.dll", dlls_folder / L"dxil.dll");
  fs::copy_file(L"dxcompiler.dll", dlls_folder / L"dxcompiler.dll");
  if (fs::exists(L"webgpu_dawn.dll")) {
    fs::copy_file(L"webgpu_dawn.dll", dlls_folder / L"webgpu_dawn.dll");
  }

  // copy the test binary to the root folder
  fs::copy_file(exe_full_path, root_folder / L"test.exe");

  // run "test.exe --test" from the test root folder
  fs::current_path(root_folder);
  return _wsystem(L"test.exe --test");
}

int run() {
  Ort::Env env{nullptr};
  int retval = 0;
  try {
    env = Ort::Env{ORT_LOGGING_LEVEL_WARNING, "Default"};

    // model is https://github.com/onnx/onnx/blob/v1.15.0/onnx/backend/test/data/node/test_abs/model.onnx
    constexpr uint8_t MODEL_DATA[] = {8, 7, 18, 12, 98, 97, 99, 107, 101, 110,
                                      100, 45, 116, 101, 115, 116, 58, 73, 10, 11,
                                      10, 1, 120, 18, 1, 121, 34, 3, 65, 98,
                                      115, 18, 8, 116, 101, 115, 116, 95, 97, 98,
                                      115, 90, 23, 10, 1, 120, 18, 18, 10, 16,
                                      8, 1, 18, 12, 10, 2, 8, 3, 10, 2,
                                      8, 4, 10, 2, 8, 5, 98, 23, 10, 1,
                                      121, 18, 18, 10, 16, 8, 1, 18, 12, 10,
                                      2, 8, 3, 10, 2, 8, 4, 10, 2, 8,
                                      5, 66, 4, 10, 0, 16, 13};

    Ort::SessionOptions session_options;
    session_options.DisableMemPattern();
    std::unordered_map<std::string, std::string> provider_options;
    session_options.AppendExecutionProvider("WebGPU", provider_options);
    Ort::Session session{env, MODEL_DATA, sizeof(MODEL_DATA), session_options};

    // successfully initialized
    std::cout << "Successfully initialized WebGPU EP." << std::endl;
    retval = 0;
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << std::endl;

    std::cerr << "Unexpected exception." << std::endl;
    retval = -1;
  }

  return retval;
}

int test_main() {
  HMODULE hModule = LoadLibraryA("dlls\\onnxruntime.dll");
  if (hModule == NULL) {
    std::cout << "Failed to load dlls\\onnxruntime.dll" << std::endl;
    std::cout << "Error code: " << GetLastError() << std::endl;
    return 1;
  }

  int retval = 0;

  using OrtGetApiBaseFunction = decltype(&OrtGetApiBase);
  auto fnOrtGetApiBase = (OrtGetApiBaseFunction)GetProcAddress(hModule, "OrtGetApiBase");
  if (fnOrtGetApiBase == NULL) {
    std::cout << "Failed to get OrtGetApiBase" << std::endl;
    retval = 1;
    goto cleanup;
  }
  Ort::InitApi(fnOrtGetApiBase()->GetApi(ORT_API_VERSION));

  retval = run();

cleanup:
  if (hModule != NULL) {
    FreeLibrary(hModule);
  }
  return retval;
}
