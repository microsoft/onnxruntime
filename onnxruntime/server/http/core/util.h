// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <boost/beast/core.hpp>
#include <boost/beast/http/status.hpp>

#include "context.h"

namespace onnxruntime {
namespace server {

namespace beast = boost::beast;  // from <boost/beast.hpp>

// Report a failure
void ErrorHandling(beast::error_code ec, char const* what);

}  // namespace server
}  // namespace onnxruntime

