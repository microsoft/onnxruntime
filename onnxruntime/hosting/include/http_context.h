//
// Created by klein on 2/26/19.
//

#ifndef BEAST_SERVER_HTTP_CONTEXT_H
#define BEAST_SERVER_HTTP_CONTEXT_H

#include <boost/beast/http.hpp>

namespace http = boost::beast::http;  // from <boost/beast/http.hpp>

class Http_Context {
 public:
  http::request<http::string_body, http::basic_fields<std::allocator<char>>> request{};
  http::response<http::string_body> response{};

  Http_Context() = default;
};

#endif  //BEAST_SERVER_HTTP_CONTEXT_H
