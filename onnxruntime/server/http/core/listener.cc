// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "listener.h"
#include "session.h"
#include "util.h"

namespace onnxruntime {
namespace server {

namespace net = boost::asio;       // from <boost/asio.hpp>
using tcp = boost::asio::ip::tcp;  // from <boost/asio/ip/tcp.hpp>

Listener::Listener(const Routes& routes, net::io_context& ioc, const tcp::endpoint& endpoint)
    : routes_(routes), acceptor_(ioc), socket_(ioc), endpoint_(endpoint) {
}

bool Listener::Init() {
  beast::error_code ec;

  // Open the acceptor
  acceptor_.open(endpoint_.protocol(), ec);
  if (ec) {
    ErrorHandling(ec, "open");
    return false;
  }

  // Allow address reuse
  acceptor_.set_option(net::socket_base::reuse_address(true), ec);
  if (ec) {
    ErrorHandling(ec, "set_option");
    return false;
  }

  // Bind to the routes address
  acceptor_.bind(endpoint_, ec);
  if (ec) {
    ErrorHandling(ec, "bind");
    return false;
  }

  // Start listening for connections
  acceptor_.listen(
      net::socket_base::max_listen_connections, ec);
  if (ec) {
    ErrorHandling(ec, "listen");
    return false;
  }

  return true;
}

bool Listener::Run() {
  if (!acceptor_.is_open()) {
    return false;
  }
  DoAccept();

  return true;
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
}  // namespace server
}  // namespace onnxruntime