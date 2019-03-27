// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef ONNXRUNTIME_HOSTING_HTTP_HTTP_SESSION_H
#define ONNXRUNTIME_HOSTING_HTTP_HTTP_SESSION_H

#include <memory>
#include <boost/beast/version.hpp>
#include <boost/asio/bind_executor.hpp>
#include <boost/beast/core/flat_buffer.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/strand.hpp>

#include "context.h"
#include "routes.h"
#include "util.h"

namespace onnxruntime {
namespace hosting {

namespace net = boost::asio;       // from <boost/asio.hpp>
namespace beast = boost::beast;    // from <boost/beast.hpp>
using tcp = boost::asio::ip::tcp;  // from <boost/asio/ip/tcp.hpp>
namespace http = beast::http;

// An implementation of a single HTTP session
// Used by a listener to hand off the work and async write back to a socket
class HttpSession : public std::enable_shared_from_this<HttpSession> {
 public:
  HttpSession(std::shared_ptr<Routes> routes, tcp::socket socket);

  // Start the asynchronous operation
  // The entrypoint for the class
  void Run() {
    DoRead();
  }

 private:
  const std::shared_ptr<Routes> routes_;
  tcp::socket socket_;
  net::strand<net::io_context::executor_type> strand_;
  beast::flat_buffer buffer_;
  http::request<http::string_body> req_;
  std::shared_ptr<void> res_{nullptr};

  // Writes the message asynchronously back to the socket
  // Stores the pointer to the message and the class itself so that
  // They do not get destructed before the async process is finished
  // If you pass shared_from_this() are guaranteed that the life time
  // of your object will be extended to as long as the function needs it
  // Most examples in boost::asio are based on this logic
  template <class Msg>
  void Send(Msg&& msg);

  // Called after the session is finished reading the message
  // Should set the response before calling Send
  template <typename Body, typename Allocator>
  void HandleRequest(http::request<Body, http::basic_fields<Allocator>>&& req);

  // Handle the request and hand it off to the user's function
  // Execute user function, handle errors
  // HttpContext parameter can be updated here or in HandleRequest
  http::status ExecuteUserFunction(HttpContext& context);

  // Asynchronously reads the request from the socket
  void DoRead();

  // Perform error checking before handing off to HandleRequest
  void OnRead(beast::error_code ec, std::size_t bytes_transferred);

  // After writing, make the session read another request
  void OnWrite(beast::error_code ec, std::size_t bytes_transferred, bool close);

  // Close the connection
  void DoClose();
};

}  // namespace hosting
}  // namespace onnxruntime

#endif  // ONNXRUNTIME_HOSTING_HTTP_HTTP_SESSION_H
