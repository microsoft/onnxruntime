// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef ONNXRUNTIME_HOSTING_HTTP_CORE_CONTEXT_H
#define ONNXRUNTIME_HOSTING_HTTP_CORE_CONTEXT_H

// boost random is using a deprecated header in 1.69
// See: https://github.com/boostorg/random/issues/49
#define BOOST_PENDING_INTEGER_LOG2_HPP
#include <boost/integer/integer_log2.hpp>

#include <string>

#include <boost/beast/http.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/uuid/uuid_generators.hpp>

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

  std::string request_id;
  http::status error_code;
  std::string error_message;

  HttpContext() : request_id(boost::uuids::to_string(boost::uuids::random_generator()())),
                  error_code(http::status::internal_server_error),
                  error_message("An unknown server error has occurred") {}

  ~HttpContext() = default;
  HttpContext(const HttpContext&) = delete;
};

}  // namespace hosting
}  // namespace onnxruntime

#endif  // ONNXRUNTIME_HOSTING_HTTP_CONTEXT_H
