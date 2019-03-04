//
// Created by klein on 2/26/19.
//
#ifndef BEAST_SERVER_UTIL_H
#define BEAST_SERVER_UTIL_H

#include <boost/beast/core.hpp>

namespace beast = boost::beast;  // from <boost/beast.hpp>

namespace beast_server {
// Report a failure
void error_handling(beast::error_code ec, char const* what) {
  std::cerr << what << ": " << ec.message() << "\n";
}
}  // namespace beast_server
#endif  //BEAST_SERVER_UTIL_H
