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

// BEGIN: Copied from debug_heap.cpp from public Windows SDK
static size_t const no_mans_land_size = 4;

struct _CrtMemBlockHeader {
  _CrtMemBlockHeader* _block_header_next;
  _CrtMemBlockHeader* _block_header_prev;
  char const* _file_name;
  int _line_number;

  int _block_use;
  size_t _data_size;

  long _request_number;
  unsigned char _gap[no_mans_land_size];

  // Followed by:
  // unsigned char    _data[_data_size];
  // unsigned char    _another_gap[no_mans_land_size];
};
// END: Copied from debug_heap.cpp from public Windows SDK

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

#ifdef _DEBUG

void PrintCrtMemStateDetails(const _CrtMemState* state, const char* state_name) {
  std::cout << "\n--- Custom Dump for _CrtMemState: " << state_name << " ---" << std::endl;
  const char* block_type_names[] = {
      "_FREE_BLOCK   (0)",
      "_NORMAL_BLOCK (1)",  // User allocations
      "_CRT_BLOCK    (2)",  // CRT internal allocations
      "_IGNORE_BLOCK (3)",
      "_CLIENT_BLOCK (4)"
      // _MAX_BLOCKS is typically 5, so indices 0-4
  };

  bool has_differences = false;
  for (int i = 0; i < _MAX_BLOCKS; ++i) {
    // For a difference state, lCounts and lSizes represent the *net change*.
    // A positive value means more blocks/bytes in the second state than the first.
    // A negative value means fewer blocks/bytes.
    if (state->lCounts[i] != 0 || state->lSizes[i] != 0) {
      has_differences = true;
      std::cout << "  Block Type " << (i < 5 ? block_type_names[i] : "UNKNOWN") << ":" << std::endl;
      std::cout << "    Net Change in Count of Blocks: " << state->lCounts[i] << std::endl;
      std::cout << "    Net Change in Total Bytes:     " << state->lSizes[i] << std::endl;
    }
  }

  if (!has_differences && (state->lHighWaterCount != 0 || state->lTotalCount != 0)) {
    // This case might occur if allocations and deallocations perfectly balanced out
    // for each block type, but there was still activity.
    std::cout << "  No net change in counts/sizes per block type, but other diffs exist." << std::endl;
  } else if (!has_differences && state->lHighWaterCount == 0 && state->lTotalCount == 0) {
    std::cout << "  No differences found in lCounts, lSizes, lHighWaterCount, or lTotalCount for this diff state." << std::endl;
  }

  // For a difference state:
  // lHighWaterCount is the maximum increase in bytes allocated at any one time between the two snapshots.
  // lTotalCount is the net increase in total bytes allocated.
  std::cout << "  lHighWaterCount (max increase in bytes between snapshots): " << state->lHighWaterCount << std::endl;
  std::cout << "  lTotalCount (net increase in total bytes allocated):     " << state->lTotalCount << std::endl;
  std::cout << "----------------------------------------------------" << std::endl;
}
// HEAP_DEBUG: Function to dump details of currently allocated _NORMAL_BLOCKs in a given memory state
void DumpCurrentlyAllocatedNormalBlocks(const _CrtMemState* memState, const char* stateName) {
  std::cout << "\n--- Details of _NORMAL_BLOCKs in State: " << stateName << " ---" << std::endl;
  _CrtMemBlockHeader* pHead;  // Pointer to the header of a memory block
  size_t totalNormalBlockBytes = 0;
  int normalBlockCount = 0;

  // memState->pBlockHeader points to the head of the linked list of memory blocks
  // that were allocated at the time this memState snapshot was taken.
  // We iterate through this list.
  for (pHead = memState->pBlockHeader; pHead != NULL; pHead = pHead->_block_header_next) {
    // We are interested in _NORMAL_BLOCKs as these are typical user allocations
    if (pHead->_block_use == _NORMAL_BLOCK) {
      normalBlockCount++;
      totalNormalBlockBytes += pHead->_data_size;  // nDataSize is the actual requested size

      std::cout << "  Allocation Request #" << pHead->_request_number << ":" << std::endl;
      if (pHead->_file_name != NULL) {
        std::cout << "    File: " << pHead->_file_name << std::endl;
      } else {
        std::cout << "    File: (N/A or not recorded)" << std::endl;
      }
      std::cout << "    Line: " << pHead->_line_number << std::endl;
      std::cout << "    Size: " << pHead->_data_size << " bytes" << std::endl;
    }
  }

  if (normalBlockCount == 0) {
    std::cout << "  No _NORMAL_BLOCKs found currently allocated in this memory state." << std::endl;
  } else {
    std::cout << "  Summary for " << stateName << ": Found " << normalBlockCount
              << " _NORMAL_BLOCKs, totaling " << totalNormalBlockBytes << " bytes." << std::endl;
  }
  std::cout << "----------------------------------------------------------" << std::endl;
}
#endif

int main() {
#ifdef _WIN32
#if defined(_DEBUG) && !defined(ONNXRUNTIME_ENABLE_MEMLEAK_CHECK)
  int tmpFlag = _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG);
  tmpFlag |= _CRTDBG_LEAK_CHECK_DF;
  tmpFlag |= _CRTDBG_ALLOC_MEM_DF;
  _CrtSetDbgFlag(tmpFlag);
  std::cout << "CRT Debug Memory Leak Detection Enabled." << std::endl;
#endif
#endif

  // HEAP_DEBUG: Declare _CrtMemState structures
#ifdef _DEBUG
  _CrtMemState s1, s2, s3_diff;
  bool heap_debug_initialized = false;
#endif

  HMODULE ort_library_handle = nullptr;
  const OrtApi* g_ort_api_instance = nullptr;  // Raw C API pointer used to initialize Ort::Api

  std::cout << "Attempting to test ONNX Runtime dynamic load/unload..." << std::endl;

  try {
    // HEAP_DEBUG: Take initial heap snapshot BEFORE loading the DLL
#ifdef _DEBUG
    _CrtMemCheckpoint(&s1);
    heap_debug_initialized = true;
    std::cout << "HEAP_DEBUG: Initial memory checkpoint (s1) taken." << std::endl;
#endif

    // 1. Dynamically load the onnxruntime.dll
    std::cout << "Loading onnxruntime.dll..." << std::endl;
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
    // auto output_shape_info = output_tensor.GetTensorTypeAndShapeInfo(); // unused

    std::cout << "Output tensor values: " << output_array[0] << ", " << output_array[1] << std::endl;
    std::cout << "Output verified successfully." << std::endl;

    std::cout << "ONNX Runtime C++ objects (session, env, etc.) are about to go out of scope and be released." << std::endl;
    // You may uncomment the following line to check if memory leak checker still works as expected.
    // env.release(); // Note: Ort::Env's destructor will handle release if not explicitly called.
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
    // TODO: take a heap snapshot before unloading the DLL so that we could have symbol information
    BOOL free_result = FreeLibrary(ort_library_handle);
    HMODULE temp_handle_before_nullptr = ort_library_handle;  // For checking GetModuleHandle below
    ort_library_handle = nullptr;
    g_ort_api_instance = nullptr;  // This pointer is now invalid
                                   // The static pointer inside Ort::Api is also now invalid.

    if (free_result) {
      std::cout << "FreeLibrary call for onnxruntime.dll succeeded." << std::endl;
    } else {
      DWORD error_code = GetLastError();
      std::cerr << "FreeLibrary call for onnxruntime.dll failed. Error code: " << error_code << std::endl;
    }

    // HEAP_DEBUG: Take second heap snapshot AFTER unloading the DLL
#ifdef _DEBUG
    if (heap_debug_initialized) {
      std::cout << "HEAP_DEBUG: Taking memory checkpoint (s2) after FreeLibrary." << std::endl;
      _CrtMemCheckpoint(&s2);

      // HEAP_DEBUG: Compare snapshots and dump statistics of the differences
      if (_CrtMemDifference(&s3_diff, &s1, &s2)) {
        // TODO: now even we have the pointers, we do not know what they were used for since the onnxruntime.dll is already gone
        std::cout << "\n---------- HEAP DIFFERENCE (s2 - s1) ----------" << std::endl;
        std::cout << "This shows memory allocated after s1 that was still allocated at s2." << std::endl;
        _CrtMemDumpStatistics(&s3_diff);
        std::cout << "\nHEAP_DEBUG: Dumping s3_diff with custom PrintCrtMemStateDetails:" << std::endl;
        PrintCrtMemStateDetails(&s3_diff, "s3_diff (Net Change: s2 - s1)");
        // NOW, if s3_diff indicates a leak of _NORMAL_BLOCKs, dump them from s2:
        if (s3_diff.lCounts[_NORMAL_BLOCK] > 0 || s3_diff.lSizes[_NORMAL_BLOCK] > 0) {
          std::cout << "\nHEAP_DEBUG: s3_diff indicates an increase in _NORMAL_BLOCKs." << std::endl;
          std::cout << "HEAP_DEBUG: Dumping all _NORMAL_BLOCKs still present in state s2 (after FreeLibrary):" << std::endl;
          DumpCurrentlyAllocatedNormalBlocks(&s2, "s2 (After Unload)");
        } else {
          std::cout << "\nHEAP_DEBUG: s3_diff did not show an increase in _NORMAL_BLOCKs." << std::endl;
        }
      } else {
        std::cout << "\nHEAP_DEBUG: No memory difference detected between s1 and s2." << std::endl;
      }
    }
#endif

    // 10.a Check if DLL is truly unloaded from the process
    std::cout << "Verifying if onnxruntime.dll is unloaded from the current process..." << std::endl;
    // Important: Use the same string as LoadLibrary for GetModuleHandle
    HMODULE module_check_handle = GetModuleHandle(TEXT("onnxruntime.dll"));
    if (module_check_handle == NULL) {
      std::cout << "onnxruntime.dll is no longer loaded in the process (GetModuleHandle returned NULL)." << std::endl;
    } else {
      // This can happen if the DLL's reference count didn't drop to zero.
      // For example, if another part of the program (or another DLL) also loaded it,
      // or if the DLL itself spawned a thread that hasn't exited and holds a reference.
      std::cout << "WARNING: onnxruntime.dll appears to STILL be loaded in the process (GetModuleHandle returned "
                << module_check_handle << ")." << std::endl;
      std::cout << "         Original handle was: " << temp_handle_before_nullptr << std::endl;
      std::cout << "         This could happen if other references to the DLL exist or if FreeLibrary didn't fully succeed due to pending operations." << std::endl;
    }
  } else {
    std::cout << "onnxruntime.dll was not loaded, skipping unload and heap diff." << std::endl;
  }

  std::cout << "Program finished." << std::endl;
  return 0;
}