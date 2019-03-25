// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include "re2/re2.h"

#include "context.h"
#include "routes.h"

namespace onnxruntime {
namespace hosting {

namespace http = boost::beast::http;  // from <boost/beast/http.hpp>

using handler_fn = std::function<void(std::string, std::string, std::string, HttpContext&)>;

bool Routes::RegisterController(http::verb method, const std::string& url_pattern, const handler_fn& controller) {
  switch (method) {
    case http::verb::get:
      this->get_fn_table.emplace_back(url_pattern, controller);
      return true;
    case http::verb::post:
      this->post_fn_table.emplace_back(url_pattern, controller);
      return true;
    default:
      return false;
  }
}

http::status Routes::ParseUrl(http::verb method,
                              const std::string& url,
                              /* out */ std::string& model_name,
                              /* out */ std::string& model_version,
                              /* out */ std::string& action,
                              /* out */ handler_fn& func) {
  std::vector<std::pair<std::string, handler_fn>> func_table;
  switch (method) {
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
    return http::status::method_not_allowed;
  }

  bool found_match = false;
  for (const auto& pattern : func_table) {
    if (re2::RE2::FullMatch(url, pattern.first, &model_name, &model_version, &action)) {
      func = pattern.second;

      found_match = true;
      break;
    }
  }

  if (!found_match) {
    return http::status::not_found;
  }

  return http::status::ok;
}

}  //namespace hosting
}  // namespace onnxruntime