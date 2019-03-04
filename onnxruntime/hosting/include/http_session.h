// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef BEAST_SERVER_HTTP_SESSION_H
#define BEAST_SERVER_HTTP_SESSION_H

#include <memory>
#include <boost/beast/version.hpp>
#include <boost/asio/bind_executor.hpp>
#include <boost/beast/core/flat_buffer.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/strand.hpp>

#include "http_context.h"
#include "routes.h"
#include "util.h"

namespace net = boost::asio;       // from <boost/asio.hpp>
namespace beast = boost::beast;    // from <boost/beast.hpp>
using tcp = boost::asio::ip::tcp;  // from <boost/asio/ip/tcp.hpp>
using handler_fn = std::function<void(std::string, std::string, std::string, Http_Context&)>;

namespace onnxruntime {

class http_session : public std::enable_shared_from_this<http_session> {
 private:
  const std::shared_ptr<Routes> routes_;
  tcp::socket socket_;
  net::strand<net::io_context::executor_type> strand_;
  beast::flat_buffer buffer_;
  http::request<http::string_body> req_;
  std::shared_ptr<void> res_{nullptr};

 public:
  explicit http_session(
      std::shared_ptr<Routes> routes,
      tcp::socket socket)
      : routes_(std::move(routes)), socket_(std::move(socket)), strand_(socket_.get_executor()) {
  }

  template <class Msg>
  void send(Msg&& msg) {
    using item_type = std::remove_reference_t<decltype(msg)>;

    auto ptr = std::make_shared<item_type>(std::move(msg));
    auto self_ = shared_from_this();
    self_->res_ = ptr;

    http::async_write(self_->socket_, *ptr,
                      net::bind_executor(strand_,
                                         [self_, close = ptr->need_eof()](beast::error_code ec, std::size_t bytes) {
                                           self_->on_write(ec, bytes, close);
                                         }));
  }

  template <typename Body, typename Allocator>
  void handle_request(http::request<Body, http::basic_fields<Allocator>>&& req) {
    Http_Context context{};
    context.request = req;

    std::string path = req.target().to_string();
    std::string model_name;
    std::string model_version;
    std::string action;
    handler_fn func;
    http::status status = routes_->parse_url(req.method(), path, model_name, model_version, action, func);

    if (http::status::ok == status) {
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

    return send(std::move(context.response));
  }

  // Start the asynchronous operation
  void run() {
    do_read();
  }

  void do_read() {
    // Make the request empty before reading,
    // otherwise the operation behavior is undefined.
    req_ = {};

    // Read a request
    http::async_read(socket_, buffer_, req_,
                     net::bind_executor(
                         strand_,
                         std::bind(
                             &http_session::on_read,
                             shared_from_this(),
                             std::placeholders::_1,
                             std::placeholders::_2)));
  }

  void on_read(
      beast::error_code ec,
      std::size_t bytes_transferred) {
    boost::ignore_unused(bytes_transferred);

    // This means they closed the connection
    if (ec == http::error::end_of_stream) {
      return do_close();
    }

    if (ec) {
      beast_server::error_handling(ec, "read");
      return;
    }

    // Send the response
    handle_request(std::move(req_));
  }

  void on_write(
      beast::error_code ec,
      std::size_t bytes_transferred,
      bool close) {
    boost::ignore_unused(bytes_transferred);

    if (ec) {
      beast_server::error_handling(ec, "write");
      return;
    }

    if (close) {
      // This means we should close the connection, usually because
      // the response indicated the "Connection: close" semantic.
      return do_close();
    }

    // We're done with the response so delete it
    res_ = nullptr;

    // Read another request
    do_read();
  }

  void do_close() {
    // Send a TCP shutdown
    beast::error_code ec;
    socket_.shutdown(tcp::socket::shutdown_send, ec);

    // At this point the connection is closed gracefully
  }
};

} // namespace onnxruntime

#endif  //BEAST_SERVER_HTTP_SESSION_H
