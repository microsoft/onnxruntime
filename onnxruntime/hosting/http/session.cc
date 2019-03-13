// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "session.h"

namespace onnxruntime {
namespace hosting {

namespace net = boost::asio;       // from <boost/asio.hpp>
namespace beast = boost::beast;    // from <boost/beast.hpp>
using tcp = boost::asio::ip::tcp;  // from <boost/asio/ip/tcp.hpp>

using handler_fn = std::function<void(std::string, std::string, std::string, HttpContext&)>;

void HttpSession::DoRead() {
  // Make the request empty before reading,
  // otherwise the operation behavior is undefined.
  req_ = {};

  http::async_read(socket_, buffer_, req_,
                   net::bind_executor(
                       strand_,
                       std::bind(
                           &HttpSession::OnRead,
                           shared_from_this(),
                           std::placeholders::_1,
                           std::placeholders::_2)));
}

void HttpSession::OnRead(beast::error_code ec, std::size_t bytes_transferred) {
  boost::ignore_unused(bytes_transferred);

  // This means they closed the connection
  if (ec == http::error::end_of_stream) {
    return DoClose();
  }

  if (ec) {
    ErrorHandling(ec, "read");
    return;
  }

  // Send the response
  HandleRequest(std::move(req_));
}

void HttpSession::OnWrite(beast::error_code ec, std::size_t bytes_transferred, bool close) {
  boost::ignore_unused(bytes_transferred);

  if (ec) {
    ErrorHandling(ec, "write");
    return;
  }

  if (close) {
    // This means we should close the connection, usually because
    // the response indicated the "Connection: close" semantic.
    return DoClose();
  }

  // We're done with the response so delete it
  res_ = nullptr;

  // Read another request
  DoRead();
}

void HttpSession::DoClose() {
  // Send a TCP shutdown
  beast::error_code ec;
  socket_.shutdown(tcp::socket::shutdown_send, ec);

  // At this point the connection is closed gracefully
}
}  // namespace hosting
}  // namespace onnxruntime