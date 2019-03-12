// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef ONNXRUNTIME_HOSTING_HTTP_LISTENER_H
#define ONNXRUNTIME_HOSTING_HTTP_LISTENER_H

#include <memory>

#include <boost/asio/ip/tcp.hpp>
#include "routes.h"
#include "util.h"

namespace onnxruntime {
namespace hosting {

namespace net = boost::asio;       // from <boost/asio.hpp>
using tcp = boost::asio::ip::tcp;  // from <boost/asio/ip/tcp.hpp>

// Listens on a socket and creates an HTTP session
class Listener : public std::enable_shared_from_this<Listener> {
  const std::shared_ptr<Routes> routes_;
  tcp::acceptor acceptor_;
  tcp::socket socket_;

 public:
  Listener(std::shared_ptr<Routes> routes, net::io_context& ioc, const tcp::endpoint& endpoint)
      : routes_(std::move(routes)), acceptor_(ioc), socket_(ioc) {
    beast::error_code ec;

    // Open the acceptor
    acceptor_.open(endpoint.protocol(), ec);
    if (ec) {
      ErrorHandling(ec, "open");
      return;
    }

    // Allow address reuse
    acceptor_.set_option(net::socket_base::reuse_address(true), ec);
    if (ec) {
      ErrorHandling(ec, "set_option");
      return;
    }

    // Bind to the routes address
    acceptor_.bind(endpoint, ec);
    if (ec) {
      ErrorHandling(ec, "bind");
      return;
    }

    // Start listening for connections
    acceptor_.listen(
        net::socket_base::max_listen_connections, ec);
    if (ec) {
      ErrorHandling(ec, "listen");
      return;
    }
  }

  // Start accepting incoming connections
  void Run() {
    if (!acceptor_.is_open()) {
      return;
    }
    DoAccept();
  }

  // Asynchronously accepts the socket
  void DoAccept() {
    acceptor_.async_accept(
        socket_,
        std::bind(
            &Listener::OnAccept,
            shared_from_this(),
            std::placeholders::_1));
  }

  // Creates the HTTP session and runs it
  void OnAccept(beast::error_code ec) {
    if (ec) {
      ErrorHandling(ec, "accept");
    } else {
      std::make_shared<HttpSession>(routes_, std::move(socket_))->Run();
    }

    // Accept another connection
    DoAccept();
  }
};

}  // namespace hosting
}  // namespace onnxruntime

#endif  // ONNXRUNTIME_HOSTING_HTTP_LISTENER_H
