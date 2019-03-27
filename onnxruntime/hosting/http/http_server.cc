// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <functional>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>
#include <boost/asio.hpp>
#include <boost/beast/http.hpp>

#include "context.h"
#include "session.h"
#include "listener.h"

#include "http_server.h"

namespace http = boost::beast::http;  // from <boost/beast/http.hpp>
namespace net = boost::asio;          // from <boost/asio.hpp>
using tcp = boost::asio::ip::tcp;     // from <boost/asio/ip/tcp.hpp>

namespace onnxruntime {
namespace hosting {

App::App() {
  http_details.address = boost::asio::ip::make_address_v4("0.0.0.0");
  http_details.port = 8001;
  http_details.threads = std::thread::hardware_concurrency();
}

App& App::Bind(net::ip::address address, unsigned short port) {
  http_details.address = std::move(address);
  http_details.port = port;
  return *this;
}

App& App::NumThreads(int threads) {
  http_details.threads = threads;
  return *this;
}

App& App::RegisterStartup(const StartFn& on_start) {
  on_start_ = on_start;
  return *this;
}

App& App::RegisterPost(const std::string& route, const HandlerFn& fn) {
  routes_->RegisterController(http::verb::post, route, fn);
  return *this;
}

App& App::RegisterError(const ErrorFn& fn) {
  routes_->RegisterErrorCallback(fn);
  return *this;
}

App& App::Run() {
  net::io_context ioc{http_details.threads};
  // Create and launch a listening port
  std::make_shared<Listener>(routes_, ioc, tcp::endpoint{http_details.address, http_details.port})->Run();

  // Run user on start function
  on_start_(http_details);

  // Run the I/O service on the requested number of threads
  std::vector<std::thread> v;
  v.reserve(http_details.threads - 1);
  for (auto i = http_details.threads - 1; i > 0; --i) {
    v.emplace_back(
        [&ioc] {
          ioc.run();
        });
  }
  ioc.run();
  return *this;
}
}  // namespace hosting
}  // namespace onnxruntime
