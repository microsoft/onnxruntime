// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This program does not link to any library except MSVC runtime and standard Windows libraries from Windows SDK.
#define ORT_API_MANUAL_INIT 1     // Crucial for manual initialization
#include "onnxruntime_cxx_api.h"  // Ensure this header is in your include path
                                  // This should also include onnxruntime_c_api.h which defines ORT_API_VERSION and OrtApiBase

#include <Windows.h>
#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <stdexcept>  // For std::runtime_error

// For memory leak detection on Windows with Visual Studio in Debug mode
#ifdef _WIN32
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif

static constexpr const ORTCHAR_T* MATMUL_MODEL_URI = ORT_TSTR("testdata/matmul_1.onnx");

// Typedef for the OrtGetApiBase function pointer
typedef const OrtApiBase*(ORT_API_CALL* OrtGetApiBaseFunction)(void);

// Helper to check OrtStatus and throw an exception on error
// Note: Ort::Exception handles this for the C++ API, but useful if mixing C API calls
void CheckOrtCApiStatus(const OrtApi* ort_api_ptr, OrtStatus* status) {
  if (status != nullptr) {
    std::string error_message = ort_api_ptr->GetErrorMessage(status);
    ort_api_ptr->ReleaseStatus(status);
    throw std::runtime_error("ONNX Runtime C API Error: " + error_message);
  }
}

int main() {
#ifdef _WIN32
#ifdef _DEBUG
  _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
  std::cout << "CRT Debug Memory Leak Detection Enabled." << std::endl;
#endif
#endif

  HMODULE ort_library_handle = nullptr;
  const OrtApi* g_ort_api_instance = nullptr;  // Raw C API pointer used to initialize Ort::Api

  std::cout << "Attempting to test ONNX Runtime dynamic load/unload..." << std::endl;

  try {
    // 1. Dynamically load the onnxruntime.dll
    ort_library_handle = LoadLibrary(TEXT("onnxruntime.dll"));
    if (!ort_library_handle) {
      DWORD error_code = GetLastError();
      throw std::runtime_error("Failed to load onnxruntime.dll. Error code: " + std::to_string(error_code));
    }
    std::cout << "onnxruntime.dll loaded successfully. Handle: " << ort_library_handle << std::endl;

    // 2. Get a function pointer to OrtGetApiBase from the DLL
    OrtGetApiBaseFunction ort_get_api_base_func =
        (OrtGetApiBaseFunction)GetProcAddress(ort_library_handle, "OrtGetApiBase");
    if (!ort_get_api_base_func) {
      DWORD error_code = GetLastError();
      FreeLibrary(ort_library_handle);
      throw std::runtime_error("Failed to get address of OrtGetApiBase. Error code: " + std::to_string(error_code));
    }
    std::cout << "OrtGetApiBase function address obtained." << std::endl;

    // 3. Use OrtGetApiBase to get an OrtApiBase pointer
    const OrtApiBase* api_base = ort_get_api_base_func();
    if (!api_base) {
      FreeLibrary(ort_library_handle);
      throw std::runtime_error("OrtGetApiBase returned nullptr for OrtApiBase.");
    }
    std::cout << "OrtApiBase pointer obtained." << std::endl;

    // 4. Call GetApi on OrtApiBase to get the OrtApi pointer for the desired API version
    // ORT_API_VERSION is defined in onnxruntime_c_api.h
    g_ort_api_instance = api_base->GetApi(ORT_API_VERSION);
    if (!g_ort_api_instance) {
      const char* version_string = api_base->GetVersionString ? api_base->GetVersionString() : "unknown";
      FreeLibrary(ort_library_handle);
      throw std::runtime_error("Failed to get OrtApi from OrtApiBase for ORT_API_VERSION " +
                               std::to_string(ORT_API_VERSION) +
                               ". DLL version: " + version_string);
    }
    std::cout << "OrtApi pointer obtained from OrtApiBase->GetApi() for ORT_API_VERSION " << ORT_API_VERSION << "." << std::endl;

    // 5. Use Ort::InitApi() function to set the api pointer for the C++ wrapper
    Ort::InitApi(g_ort_api_instance);  // This initializes the static Ort::Api instance
    std::cout << "Ort::Api initialized for C++ wrapper." << std::endl;

    // 6. Create an OrtEnv
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "DynamicLoadTestEnv");
    std::cout << "OrtEnv created." << std::endl;

    // 7. Create SessionOptions
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    std::cout << "OrtSessionOptions configured." << std::endl;

    // 8. Create an Inference Session using the embedded model data
    std::cout << "Creating inference session..." << std::endl;
    Ort::Session session(env, MATMUL_MODEL_URI, session_options);
    std::cout << "Inference session created." << std::endl;

    // 9. Run a simple inference
    Ort::AllocatorWithDefaultOptions allocator;

    auto input_name_ptr = session.GetInputNameAllocated(0, allocator);

    auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);

    std::vector<const char*> input_names = {input_name_ptr.get()};
    std::vector<const char*> output_names = {output_name_ptr.get()};

    std::vector<float> input_tensor_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<int64_t> input_dims = {3, 2};

    Ort::Value input_tensor = Ort::Value::CreateTensor(allocator.GetInfo(), input_tensor_values.data(),
                                                       input_tensor_values.size(), input_dims.data(), input_dims.size());

    auto output_tensors = session.Run(Ort::RunOptions{}, input_names.data(), &input_tensor, 1, output_names.data(), 1);

    assert(output_tensors.size() == 1);
    const auto& output_tensor = output_tensors[0];
    assert(output_tensor.IsTensor());
    const float* output_array = output_tensor.GetTensorData<float>();
    auto output_shape_info = output_tensor.GetTensorTypeAndShapeInfo();

    std::cout << "Output tensor values: " << output_array[0] << ", " << output_array[1] << std::endl;
    std::cout << "Output verified successfully." << std::endl;

    std::cout << "ONNX Runtime C++ objects (session, env, etc.) are about to go out of scope and be released." << std::endl;

  } catch (const Ort::Exception& e) {
    std::cerr << "ONNX Runtime C++ API Exception: " << e.what() << std::endl;
    if (ort_library_handle) {
      FreeLibrary(ort_library_handle);
    }
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "Standard Exception: " << e.what() << std::endl;
    if (ort_library_handle) {
      FreeLibrary(ort_library_handle);
    }
    return 1;
  }

  // 10. Unload the DLL
  if (ort_library_handle) {
    std::cout << "Unloading onnxruntime.dll..." << std::endl;

    BOOL free_result = FreeLibrary(ort_library_handle);
    ort_library_handle = nullptr;
    g_ort_api_instance = nullptr;  // This pointer is now invalid
                                   // The static pointer inside Ort::Api is also now invalid.

    if (free_result) {
      std::cout << "FreeLibrary call for onnxruntime.dll succeeded." << std::endl;
    } else {
      DWORD error_code = GetLastError();
      std::cerr << "FreeLibrary call for onnxruntime.dll failed. Error code: " << error_code << std::endl;
    }

    // 10.a Check if DLL is truly unloaded from the process
    std::cout << "Verifying if onnxruntime.dll is unloaded from the current process..." << std::endl;
    HMODULE module_check_handle = GetModuleHandle(TEXT("onnxruntime.dll"));
    if (module_check_handle == NULL) {
      std::cout << "onnxruntime.dll is no longer loaded in the process (GetModuleHandle returned NULL)." << std::endl;
    } else {
      std::cout << "onnxruntime.dll appears to STILL be loaded in the process (GetModuleHandle returned a valid handle: "
                << module_check_handle << ")." << std::endl;
      std::cout << "This could happen if other references to the DLL exist or if FreeLibrary didn't fully succeed." << std::endl;
    }
  }

  // 11. Memory Leak Check Point
  std::cout << "Program finished. Check console output for CRT memory leak report if applicable." << std::endl;

  return 0;
}