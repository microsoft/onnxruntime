// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

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
namespace server {

namespace http = beast::http;      // from <boost/beast/http.hpp>
namespace net = boost::asio;       // from <boost/asio.hpp>
using tcp = boost::asio::ip::tcp;  // from <boost/asio/ip/tcp.hpp>

struct Details {
  net::ip::address address;
  unsigned short port;
  int threads;
};

using StartFn = std::function<void(Details&)>;

// Accepts incoming connections and launches the sessions
// Each method returns the app itself so methods can be chained
class App {
 public:
  App();

  App& Bind(net::ip::address address, unsigned short port);
  App& NumThreads(int threads);
  App& RegisterStartup(const StartFn& fn);
  App& RegisterPost(const std::string& route, const HandlerFn& fn);
  App& RegisterError(const ErrorFn& fn);
  App& Run();

 private:
  Routes routes_{};
  StartFn on_start_ = {};
  Details http_details{};
};
}  // namespace server
}  // namespace onnxruntime
