// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "session.h"

namespace onnxruntime {
namespace hosting {

namespace net = boost::asio;       // from <boost/asio.hpp>
namespace beast = boost::beast;    // from <boost/beast.hpp>
using tcp = boost::asio::ip::tcp;  // from <boost/asio/ip/tcp.hpp>

using handler_fn = std::function<void(std::string, std::string, std::string, HttpContext&)>;

HttpSession::HttpSession(std::shared_ptr<Routes> routes, tcp::socket socket)
    : routes_(std::move(routes)), socket_(std::move(socket)), strand_(socket_.get_executor()) {
}

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

template <class Msg>
void HttpSession::Send(Msg&& msg) {
  using item_type = std::remove_reference_t<decltype(msg)>;

  auto ptr = std::make_shared<item_type>(std::move(msg));
  auto self_ = shared_from_this();
  self_->res_ = ptr;

  http::async_write(self_->socket_, *ptr,
                    net::bind_executor(strand_,
                                       [self_, close = ptr->need_eof()](beast::error_code ec, std::size_t bytes) {
                                         self_->OnWrite(ec, bytes, close);
                                       }));
}

template <typename Body, typename Allocator>
void HttpSession::HandleRequest(boost::beast::http::request<Body, boost::beast::http::basic_fields<Allocator> >&& req) {
  HttpContext context{};
  context.request = req;

  std::string path = req.target().to_string();
  std::string model_name;
  std::string model_version;
  std::string action;
  handler_fn func;
  http::status status = routes_->ParseUrl(req.method(), path, model_name, model_version, action, func);

  if (http::status::ok == status && func != nullptr) {
    func(model_name, model_version, action, context);
  } else {
    http::response<http::string_body> res{status, req.version()};
    res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
    res.set(http::field::content_type, "text/plain");
    res.keep_alive(req.keep_alive());
    res.body() = std::string("Something failed\n");
    res.prepare_payload();
    context.response = res;
  }

  return Send(std::move(context.response));
}

}  // namespace hosting
}  // namespace onnxruntime