#pragma once

#include <string>

namespace onnxruntime {
namespace server {
namespace util {
std::string InternalRequestId();
extern const std::string MS_REQUEST_ID_HEADER;
extern const std::string MS_CLIENT_REQUEST_ID_HEADER;
}  // namespace util
}  // namespace server
}  // namespace onnxruntime