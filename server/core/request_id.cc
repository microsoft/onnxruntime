// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "request_id.h"
// boost random is using a deprecated header in 1.69
// See: https://github.com/boostorg/random/issues/49
#define BOOST_PENDING_INTEGER_LOG2_HPP
#include <boost/integer/integer_log2.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

namespace onnxruntime {
namespace server {
namespace util {
std::string InternalRequestId() {
  return boost::uuids::to_string(boost::uuids::random_generator()());
}
const std::string MS_REQUEST_ID_HEADER = "x-ms-request-id";
const std::string MS_CLIENT_REQUEST_ID_HEADER = "x-ms-client-request-id";
}  // namespace util
}  // namespace server
}  // namespace onnxruntime