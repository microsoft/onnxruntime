// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "http_server.h"
#include "json_handling.h"

namespace onnxruntime {
namespace hosting {

namespace beast = boost::beast;
namespace http = beast::http;

void BadRequest(HttpContext& context, const std::string& error_message);

// TODO: decide whether this should be a class
void Predict(const std::string& name,
             const std::string& version,
             const std::string& action,
             /* in, out */ HttpContext& context,
             HostingEnvironment& env);

}  // namespace hosting
}  // namespace onnxruntime
