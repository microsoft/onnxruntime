#pragma once

#include <napi.h>

namespace Ort {
struct SessionOptions;
}

// parse a Javascript session options object and fill the native SessionOptions object.
void ParseSessionOptions(const Napi::Object options, Ort::SessionOptions &sessionOptions);
