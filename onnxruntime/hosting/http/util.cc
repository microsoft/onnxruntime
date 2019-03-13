// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include <boost/beast/core.hpp>
#include "util.h"

namespace onnxruntime {
namespace hosting {

// Report a failure
void ErrorHandling(beast::error_code ec, char const* what) {
  std::cerr << what << ": " << ec.message() << "\n";
}
}  // namespace hosting
}  // namespace onnxruntime