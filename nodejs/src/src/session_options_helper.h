#ifndef ONNXRUNTIME_NODE_SESSION_OPTIONS_HELPER_H
#define ONNXRUNTIME_NODE_SESSION_OPTIONS_HELPER_H

#pragma once

#include <napi.h>

namespace Ort {
struct SessionOptions;
}

// parse a Javascript session options object and fill the native SessionOptions object.
void ParseSessionOptions(const Napi::Value value, Ort::SessionOptions &sessionOptions);

#endif
