// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <boost/beast/core.hpp>
#include <boost/beast/http/status.hpp>
#include <google/protobuf/stubs/status.h>

#include "server/http/core/context.h"

namespace onnxruntime {
namespace server {

namespace beast = boost::beast;  // from <boost/beast.hpp>

enum class SupportedContentType : int {
  Unknown,
  Json,
  PbByteArray
};

// Mapping protobuf status to http status
boost::beast::http::status GetHttpStatusCode(const google::protobuf::util::Status& status);

// "Content-Type" header field in request is MUST-HAVE.
// Currently we only support two types of input content type: application/json and application/octet-stream
SupportedContentType GetRequestContentType(const HttpContext& context);

// "Accept" header field in request is OPTIONAL.
// Currently we only support three types of response content type: */*, application/json and application/octet-stream
SupportedContentType GetResponseContentType(const HttpContext& context);

}  // namespace server
}  // namespace onnxruntime
