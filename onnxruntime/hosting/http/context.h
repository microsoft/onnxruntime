// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef ONNXRUNTIME_HOSTING_HTTP_HTTP_CONTEXT_H
#define ONNXRUNTIME_HOSTING_HTTP_HTTP_CONTEXT_H

#include <boost/beast/http.hpp>

namespace onnxruntime {
namespace hosting {

namespace http = boost::beast::http;  // from <boost/beast/http.hpp>

// This class represents the HTTP context given to the user
// Currently, we are just giving the Boost request and response object
// But in the future we should write a wrapper around them
class HttpContext {
 public:
  http::request<http::string_body, http::basic_fields<std::allocator<char>>> request{};
  http::response<http::string_body> response{};

  HttpContext() = default;
};

}  // namespace hosting
}  // namespace onnxruntime

#endif  // ONNXRUNTIME_HOSTING_HTTP_HTTP_CONTEXT_H
