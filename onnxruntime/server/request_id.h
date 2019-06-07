#pragma once

#include <string>

namespace onnxruntime {
namespace server {
namespace util {
std::string InternalRequestId();
extern const std::string REQUEST_HEADER;
extern const std::string CLIENT_REQUEST_HEADER;
}  // namespace util
}  // namespace server
}  // namespace onnxruntime