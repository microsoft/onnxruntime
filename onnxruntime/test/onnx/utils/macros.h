// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <iostream>

// When onnxruntime is built as a shared library, ORT env is created on the DLL side and same as default logger that comes with it.
// Since the public API doesn’t expose that logger to the host app, we print messages with std::cout instead.
#ifdef BUILD_SHARED_LIB
#define TEST_LOG_ERROR(...) std::cerr << __VA_ARGS__ << std::endl;
#define TEST_LOG_INFO(...) std::cout << __VA_ARGS__ << std::endl;
#define TEST_LOG_VERBOSE(...) std::cout << __VA_ARGS__ << std::endl;
#else
#define TEST_LOG_ERROR(...) LOGS_DEFAULT(ERROR) << __VA_ARGS__;
#define TEST_LOG_INFO(...) LOGS_DEFAULT(INFO) << __VA_ARGS__;
#define TEST_LOG_VERBOSE(...) LOGS_DEFAULT(VERBOSE) << __VA_ARGS__;
#endif