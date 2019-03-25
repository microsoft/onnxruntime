// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef ONNXRUNTIME_HOSTING_HTTP_ROUTES_H
#define ONNXRUNTIME_HOSTING_HTTP_ROUTES_H

#include <boost/beast/http.hpp>

#include "context.h"

namespace onnxruntime {
namespace hosting {

namespace http = boost::beast::http;  // from <boost/beast/http.hpp>

using handler_fn = std::function<void(std::string, std::string, std::string, HttpContext&)>;

// This class maintains two lists of regex -> function lists. One for POST requests and one for GET requests
// If the incoming URL could match more than one regex, the first one will win.
class Routes {
 public:
  Routes() = default;
  bool RegisterController(http::verb method, const std::string& url_pattern, const handler_fn& controller);

  http::status ParseUrl(http::verb method,
                        const std::string& url,
                        /* out */ std::string& model_name,
                        /* out */ std::string& model_version,
                        /* out */ std::string& action,
                        /* out */ handler_fn& func);

 private:
  std::vector<std::pair<std::string, handler_fn>> post_fn_table;
  std::vector<std::pair<std::string, handler_fn>> get_fn_table;
  // TODO: server error callback
};

}  //namespace hosting
}  // namespace onnxruntime

#endif  // ONNXRUNTIME_HOSTING_HTTP_ROUTES_H
