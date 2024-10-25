// Copyright (c) Microsoft Corporation. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
// Licensed under the MIT License.

#include <iostream>

#include "core/session/onnxruntime_cxx_api.h"

#include <google/protobuf/stubs/common.h>

#include "dawn/native/DawnNative.h"

#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) {
#else
int main(int argc, char* argv[]) {
#endif
  bool no_proc_table = argc > 0 &&
#ifdef _WIN32
                       wcscmp(L"--no_proc_table", argv[argc - 1]) == 0;
#else
                       strcmp("--no_proc_table", argv[argc - 1]) == 0;
#endif

  int retval = 0;
  Ort::Env env{nullptr};
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
    if (!no_proc_table) {
      provider_options["dawnProcTable"] = std::to_string(reinterpret_cast<size_t>(&dawn::native::GetProcs()));
    }
    session_options.AppendExecutionProvider("WebGPU", provider_options);
    Ort::Session session{env, MODEL_DATA, sizeof(MODEL_DATA), session_options};

    if (no_proc_table) {
      std::cerr << "DawnProcTable is not passing to ONNX Runtime, but no exception is thrown." << std::endl;
      retval = -1;
    } else {
      // successfully initialized
      std::cout << "Successfully initialized WebGPU EP." << std::endl;
      retval = 0;
    }
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << std::endl;

    if (no_proc_table) {
      std::cout << "DawnProcTable is not passing to ONNX Runtime, so an exception is thrown as expected." << std::endl;
      retval = 0;
    } else {
      std::cerr << "Unexpected exception." << std::endl;
      retval = -1;
    }
  }

  ::google::protobuf::ShutdownProtobufLibrary();
  return retval;
}
