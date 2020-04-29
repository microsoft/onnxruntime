#ifndef ONNXRUNTIME_NODE_RUN_OPTIONS_HELPER_H
#define ONNXRUNTIME_NODE_RUN_OPTIONS_HELPER_H

#pragma once

#include <napi.h>

namespace Ort {
struct RunOptions;
}

// parse a Javascript run options object and fill the native RunOptions object.
void ParseRunOptions(const Napi::Object options, Ort::RunOptions &runOptions);

#endif
