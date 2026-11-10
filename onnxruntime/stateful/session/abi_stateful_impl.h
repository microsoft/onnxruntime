// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

namespace onnxruntime {
class SessionChain;
}

struct OrtSessionChain {
  std::shared_ptr<const onnxruntime::SessionChain> value;
};
