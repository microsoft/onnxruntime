// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "listener.h"
#include "session.h"
#include "util.h"

namespace onnxruntime {
namespace hosting {

namespace net = boost::asio;       // from <boost/asio.hpp>
using tcp = boost::asio::ip::tcp;  // from <boost/asio/ip/tcp.hpp>

Listener::Listener(std::shared_ptr<Routes> routes, net::io_context& ioc, const tcp::endpoint& endpoint)
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

void Listener::Run() {
  if (!acceptor_.is_open()) {
    return;
  }
  DoAccept();
}

void Listener::DoAccept() {
  acceptor_.async_accept(
      socket_,
      std::bind(
          &Listener::OnAccept,
          shared_from_this(),
          std::placeholders::_1));
}

void Listener::OnAccept(beast::error_code ec) {
  if (ec) {
    ErrorHandling(ec, "accept");
  } else {
    std::make_shared<HttpSession>(routes_, std::move(socket_))->Run();
  }

  // Accept another connection
  DoAccept();
}
}  // namespace hosting
}  // namespace onnxruntime