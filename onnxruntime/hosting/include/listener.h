// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef BEAST_SERVER_LISTENER_H
#define BEAST_SERVER_LISTENER_H

#include <memory>

#include <boost/asio/ip/tcp.hpp>
#include "routes.h"
#include "util.h"

namespace net = boost::asio;       // from <boost/asio.hpp>
using tcp = boost::asio::ip::tcp;  // from <boost/asio/ip/tcp.hpp>

class listener : public std::enable_shared_from_this<listener> {
  const std::shared_ptr<Routes> routes_;
  tcp::acceptor acceptor_;
  tcp::socket socket_;

 public:
  listener(std::shared_ptr<Routes> routes, net::io_context& ioc, const tcp::endpoint& endpoint)
      : routes_(std::move(routes)), acceptor_(ioc), socket_(ioc) {
    beast::error_code ec;

    // Open the acceptor
    acceptor_.open(endpoint.protocol(), ec);
    if (ec) {
      beast_server::error_handling(ec, "open");
      return;
    }

    // Allow address reuse
    acceptor_.set_option(net::socket_base::reuse_address(true), ec);
    if (ec) {
      beast_server::error_handling(ec, "set_option");
      return;
    }

    // Bind to the routes address
    acceptor_.bind(endpoint, ec);
    if (ec) {
      beast_server::error_handling(ec, "bind");
      return;
    }

    // Start listening for connections
    acceptor_.listen(
        net::socket_base::max_listen_connections, ec);
    if (ec) {
      beast_server::error_handling(ec, "listen");
      return;
    }
  }

  // Start accepting incoming connections
  void run() {
    if (!acceptor_.is_open()) {
      return;
    }
    do_accept();
  }

  void do_accept() {
    acceptor_.async_accept(
        socket_,
        std::bind(
            &listener::on_accept,
            shared_from_this(),
            std::placeholders::_1));
  }

  void on_accept(beast::error_code ec) {
    if (ec) {
      beast_server::error_handling(ec, "accept");
    } else {
      // Create the session and run it
      std::make_shared<http_session>(
          routes_,
          std::move(socket_))
          ->run();
    }

    // Accept another connection
    do_accept();
  }
};

#endif  //BEAST_SERVER_LISTENER_H
