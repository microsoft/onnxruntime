// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef BEAST_SERVER_ROUTES_H
#define BEAST_SERVER_ROUTES_H

#include <regex>
#include <vector>
#include <boost/beast/http.hpp>
#include "http_context.h"

namespace http = boost::beast::http;  // from <boost/beast/http.hpp>

namespace onnxruntime {

using handler_fn = std::function<void(std::string, std::string, std::string, Http_Context&)>;

// This class maintains two lists of regex -> function lists. One for POST requests and one for GET requests
// If the incoming URL could match more than one regex, the first one will win.
class Routes {
 public:
  Routes() = default;
  bool register_controller(http::verb method, const std::regex& url_pattern, const handler_fn& controller) {
    switch(method)
    {
      case http::verb::get:
        this->get_fn_table.push_back(make_pair(url_pattern, controller));
        return true;
      case http::verb::post:
        this->post_fn_table.push_back(make_pair(url_pattern, controller));
        return true;
      default:
        return false;
    }
  }

  http::status parse_url(http::verb method,
                         const std::string& url,
                         /* out */ std::string& model_name,
                         /* out */ std::string& model_version,
                         /* out */ std::string& action,
                         /* out */ handler_fn& func) {
    std::vector<std::pair<std::regex, handler_fn>> func_table;       
    switch(method)
    {
      case http::verb::get:
        func_table = this->get_fn_table;
        break;
      case http::verb::post:
        func_table = this->post_fn_table;
        break;
      default:
        std::cout << "Unsupported method: [" << method << "]" << std::endl;
        return http::status::method_not_allowed;
    }

    if (func_table.empty()) {
      std::cout << "Unsupported method: [" << method << "]" << std::endl;
      return http::status::method_not_allowed;
    }

    std::smatch m{};
    bool found_match = false;
    for (const auto& pattern : func_table) {
      // TODO: use re2 for matching
      if (std::regex_match(url, m, pattern.first)) {
        model_name = m[1];
        model_version = m[2];
        action = m[3];
        func = pattern.second;

        found_match = true;
        break;
      }
    }

    if (!found_match) {
      std::cerr << "Path not found: [" << url << "]" << std::endl;
      return http::status::not_found;
    }

    return http::status::ok;
  }

 private:
  std::vector<std::pair<std::regex, handler_fn>> post_fn_table;
  std::vector<std::pair<std::regex, handler_fn>> get_fn_table;
};

} // namespace onnxruntime

#endif  //BEAST_SERVER_ROUTES_H