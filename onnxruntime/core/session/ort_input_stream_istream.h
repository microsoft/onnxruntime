// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <istream>
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {

class OrtInputStreamIStream : public std::istream {
 public:
  OrtInputStreamIStream(OrtInputStream& stream, size_t buffer_size);
  virtual ~OrtInputStreamIStream();
};

}  // namespace onnxruntime
