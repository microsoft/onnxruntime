// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// Test framework headers (highest compilation time impact)
#include "gtest/gtest.h"
#include "gtest/gtest-assertion-result.h"
#include "gtest/gtest-message.h"
#include "gtest/internal/gtest-port.h"

// Core test utilities (most frequently used in tests)
#include "test/providers/provider_test_utils.h"
#include "test/providers/checkers.h"

// ONNX and Protocol Buffer headers
#include "core/graph/onnx_protobuf.h"
#include "onnx/defs/schema.h"

// Data types and framework headers
#include "core/framework/data_types.h"

// Windows-specific headers (if applicable)
#ifdef _WIN32
#include <windows.h>
#endif
