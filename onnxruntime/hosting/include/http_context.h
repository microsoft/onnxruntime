// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef BEAST_SERVER_HTTP_CONTEXT_H
#define BEAST_SERVER_HTTP_CONTEXT_H

#include <boost/beast/http.hpp>

namespace http = boost::beast::http;  // from <boost/beast/http.hpp>

namespace onnxruntime {

class Http_Context {
 public:
  http::request<http::string_body, http::basic_fields<std::allocator<char>>> request{};
  http::response<http::string_body> response{};

  Http_Context() = default;
};

} // namespace onnxruntime

#endif  //BEAST_SERVER_HTTP_CONTEXT_H
