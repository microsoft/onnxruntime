// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef ONNXRUNTIME_HOSTING_HTTP_CORE_UTIL_H
#define ONNXRUNTIME_HOSTING_HTTP_CORE_UTIL_H

#include <boost/beast/core.hpp>
#include <boost/beast/http/status.hpp>

#include "context.h"

namespace onnxruntime {
namespace hosting {

namespace beast = boost::beast;  // from <boost/beast.hpp>

// Report a failure
void ErrorHandling(beast::error_code ec, char const* what);

}  // namespace hosting
}  // namespace onnxruntime

#endif  // ONNXRUNTIME_HOSTING_HTTP_UTIL_H
