// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <boost/asio.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/beast/http.hpp>

#include "context.h"
#include "session.h"
#include "listener.h"
#include "routes.h"
#include "util.h"

#include "http_server.h"

namespace http = boost::beast::http;  // from <boost/beast/http.hpp>
namespace net = boost::asio;          // from <boost/asio.hpp>
using tcp = boost::asio::ip::tcp;     // from <boost/asio/ip/tcp.hpp>

namespace onnxruntime {
namespace hosting {

using handler_fn = std::function<void(std::string, std::string, std::string, HttpContext&)>;

App::App() {
  // TODO: defaults should come from central place
  address_ = boost::asio::ip::make_address_v4("0.0.0.0");
  port_ = 8080;
  threads_ = std::thread::hardware_concurrency();
}

App& App::Bind(net::ip::address address, unsigned short port) {
  address_ = std::move(address);
  port_ = port;
  return *this;
}

App& App::NumThreads(int threads) {
  threads_ = threads;
  return *this;
}

App& App::Post(const std::string& route, const handler_fn& fn) {
  routes->RegisterController(http::verb::post, route, fn);
  return *this;
}

App& App::Run() {
  net::io_context ioc{threads_};
  // Create and launch a listening port
  std::make_shared<Listener>(routes, ioc, tcp::endpoint{address_, port_})->Run();

  // TODO: use logger
  std::cout << "Listening at: \n"
            << std::endl;
  std::cout << "\thttp://" << address_ << ":" << port_ << std::endl;

  // Run the I/O service on the requested number of threads
  std::vector<std::thread> v;
  v.reserve(threads_ - 1);
  for (auto i = threads_ - 1; i > 0; --i) {
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
