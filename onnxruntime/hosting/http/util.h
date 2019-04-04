// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef ONNXRUNTIME_HOSTING_HTTP_UTIL_H
#define ONNXRUNTIME_HOSTING_HTTP_UTIL_H

#include <boost/beast/core.hpp>
#include <boost/beast/http/status.hpp>
#include <google/protobuf/stubs/status.h>

#include "context.h"

namespace onnxruntime {
namespace hosting {

namespace beast = boost::beast;  // from <boost/beast.hpp>

enum class SupportedContentType : int {
  Unknown,
  Json,
  PbByteArray
};

// Report a failure
void ErrorHandling(beast::error_code ec, char const* what);

// Mapping protobuf status to http status
boost::beast::http::status GetHttpStatusCode(const google::protobuf::util::Status& status);

// "Content-Type" header field in request is MUST-HAVE.
// Currently we only support two types of input content type: application/json and application/octet-stream
SupportedContentType GetRequestContentType(const HttpContext& context);

// "Accept" header field in request is OPTIONAL.
// Currently we only support three types of response content type: */*, application/json and application/octet-stream
SupportedContentType GetResponseContentType(const HttpContext& context);

}  // namespace hosting
}  // namespace onnxruntime

#endif  // ONNXRUNTIME_HOSTING_HTTP_UTIL_H
