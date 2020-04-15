#ifndef ONNXRUNTIME_NODE_SESSION_OPTIONS_HELPER_H
#define ONNXRUNTIME_NODE_SESSION_OPTIONS_HELPER_H

#pragma once

#include <napi.h>

namespace Ort {
struct SessionOptions;
}

void ParseSessionOptions(const Napi::Value value, Ort::SessionOptions &sessionOptions);

#endif
