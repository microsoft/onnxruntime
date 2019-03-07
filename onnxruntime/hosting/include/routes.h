// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef BEAST_SERVER_ROUTES_H
#define BEAST_SERVER_ROUTES_H

#include <regex>
#include <unordered_map>
#include <boost/beast/http.hpp>
#include "http_context.h"

namespace http = boost::beast::http;  // from <boost/beast/http.hpp>

namespace onnxruntime {

using handler_fn = std::function<void(std::string, std::string, std::string, Http_Context&)>;

class Routes {
 public:
  Routes() = default;
  bool register_controller(http::verb method, const std::regex& url_pattern, const handler_fn& controller) {
    if (this->fn_table.find(method) == this->fn_table.end()) {
      this->fn_table[method] = std::vector<std::pair<std::regex, handler_fn>>{};
    }
    (this->fn_table[method]).push_back(make_pair(url_pattern, controller));
    return true;
  }

  http::status parse_url(http::verb method,
                         const std::string& url,
                         /* out */ std::string& model_name,
                         /* out */ std::string& model_version,
                         /* out */ std::string& action,
                         /* out */ handler_fn& func) {
    if (this->fn_table.find(method) == this->fn_table.end()) {
      std::cout << "Unsupported method: [" << method << "]" << std::endl;
      return http::status::method_not_allowed;
    }

    std::smatch m{};
    bool found_match = false;
    for (const auto& pattern : this->fn_table[method]) {
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
  std::unordered_map<http::verb, std::vector<std::pair<std::regex, handler_fn>>> fn_table;
};

} // namespace onnxruntime

#endif  //BEAST_SERVER_ROUTES_H
