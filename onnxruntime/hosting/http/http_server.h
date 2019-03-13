// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef ONNXRUNTIME_HOSTING_HTTP_BEAST_HTTP_H
#define ONNXRUNTIME_HOSTING_HTTP_BEAST_HTTP_H

#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "util.h"
#include "context.h"
#include "routes.h"
#include "session.h"
#include "listener.h"

namespace onnxruntime {
namespace hosting {

namespace http = beast::http;      // from <boost/beast/http.hpp>
namespace net = boost::asio;       // from <boost/asio.hpp>
using tcp = boost::asio::ip::tcp;  // from <boost/asio/ip/tcp.hpp>

using handler_fn = std::function<void(std::string, std::string, std::string, HttpContext&)>;

// Accepts incoming connections and launches the sessions
// Each method returns the app itself so methods can be chained
class App {
 public:
  App();

  App& Bind(net::ip::address address, unsigned short port);
  App& NumThreads(int threads);
  App& Post(const std::string& route, const handler_fn& fn);
  App& Run();

 private:
  const std::shared_ptr<Routes> routes{std::make_shared<Routes>()};
  net::ip::address address_;
  unsigned short port_;
  int threads_;
};
}  // namespace hosting
}  // namespace onnxruntime

#endif  // ONNXRUNTIME_HOSTING_HTTP_BEAST_HTTP_H
