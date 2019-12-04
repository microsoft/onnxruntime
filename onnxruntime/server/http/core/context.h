// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#include <boost/beast/http.hpp>
#include "request_id.h"

namespace onnxruntime {
namespace server {

namespace http = boost::beast::http;  // from <boost/beast/http.hpp>

// This class represents the HTTP context given to the user
// Currently, we are just giving the Boost request and response object
// But in the future we should write a wrapper around them
class HttpContext {
 public:
  http::request<http::string_body, http::basic_fields<std::allocator<char>>> request{};
  http::response<http::string_body> response{};

  const std::string request_id;
  std::string client_request_id;
  http::status error_code;
  std::string error_message;

  HttpContext() : request_id(util::InternalRequestId()),
                  client_request_id(""),
                  error_code(http::status::internal_server_error),
                  error_message("An unknown server error has occurred") {}

  ~HttpContext() = default;
  HttpContext(const HttpContext&) = delete;
};

}  // namespace server
}  // namespace onnxruntime
