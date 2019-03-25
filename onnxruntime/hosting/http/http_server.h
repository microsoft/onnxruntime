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

struct Details {
  net::ip::address address;
  unsigned short port;
  int threads;
};

using start_fn = std::function<void(Details&)>;

// Accepts incoming connections and launches the sessions
// Each method returns the app itself so methods can be chained
class App {
 public:
  App();

  App& Bind(net::ip::address address, unsigned short port);
  App& NumThreads(int threads);
  App& RegisterStartup(const start_fn& fn);
  App& RegisterPost(const std::string& route, const handler_fn& fn);
  App& Run();

 private:
  const std::shared_ptr<Routes> routes_ = std::make_shared<Routes>();
  start_fn on_start_ = {};
  Details http_details{};
};
}  // namespace hosting
}  // namespace onnxruntime

#endif  // ONNXRUNTIME_HOSTING_HTTP_BEAST_HTTP_H
