// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace invoker {

// todo: disable this when it is minimal build
void* CreateOp(void* sess, const char* domain, const char* op_name, const int& version);

}//invoker
}//onnxruntime