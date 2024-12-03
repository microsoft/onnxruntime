// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_c_api_ep.h"

namespace Ort {
namespace PluginEP {

struct Graph {
explicit Graph(const OrtGraphViewer*);
const char* GetName();
private:
const OrtGraphViewer* graph_;
};

struct Node {
explicit Node(const OrtNode*);
const char* GetName();
private:
const OrtNode* node_;
};

}
}

#include "onnxruntime_cxx_inline_ep.h"
