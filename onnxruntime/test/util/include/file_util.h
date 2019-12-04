// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "core/session/onnxruntime_c_api.h"
#include <stdio.h>
#include <string>

namespace onnxruntime {
namespace test {
void CreateTestFile(FILE*& out, std::basic_string<ORTCHAR_T>& filename_template);
void CreateTestFile(int& out, std::basic_string<ORTCHAR_T>& filename_template);
void DeleteFileFromDisk(const ORTCHAR_T* path);

}  // namespace test
}  // namespace onnxruntime