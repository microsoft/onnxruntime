// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// Extends config_key for testing purpose only, see following files for more information:
// - include/onnxruntime/core/session/onnxruntime_run_options_config_keys.h
// - include/onnxruntime/core/framework/run_options.h
// - onnxruntime/core/framework/config_options.h

// Key for enabling OpTester for additionally test an OpKernel with EP config to enable TunableOp. Valid values are
// "true" or "false"
static const char* const kOpTesterRunOptionsConfigTestTunableOp = "op_tester.is_tunable_op_under_test";
