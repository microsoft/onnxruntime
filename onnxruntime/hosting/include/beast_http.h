// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "util.h"
#include "http_context.h"
#include "routes.h"
#include "http_session.h"
#include "listener.h"

namespace http = beast::http;      // from <boost/beast/http.hpp>
namespace net = boost::asio;       // from <boost/asio.hpp>
using tcp = boost::asio::ip::tcp;  // from <boost/asio/ip/tcp.hpp>

namespace onnxruntime {

using handler_fn = std::function<void(std::string, std::string, std::string, Http_Context&)>;

// Accepts incoming connections and launches the sessions
class App {
 public:
  App() {
    // TODO: defaults should come from central place
    address_ = boost::asio::ip::make_address_v4("0.0.0.0");
    port_ = 8080;
    threads_ = std::thread::hardware_concurrency();
  }

  App& bind(net::ip::address address, unsigned short port) {
    address_ = std::move(address);
    port_ = port;
    return *this;
  }

  App& num_threads(int threads) {
    threads_ = threads;
    return *this;
  }

  App& post(const std::string& route, handler_fn fn) {
    //    routes->http_posts[route] = std::move(fn);
    routes->register_controller(http::verb::post, route, fn);
    return *this;
  }

  App& run() {
    net::io_context ioc{threads_};
    // Create and launch a listening port
    std::make_shared<listener>(routes, ioc, tcp::endpoint{address_, port_})->run();

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

 private:
  const std::shared_ptr<Routes> routes{std::make_shared<Routes>()};
  net::ip::address address_;
  unsigned short port_;
  int threads_;
};

} // namespace onnxruntime


