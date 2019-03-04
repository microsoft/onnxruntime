// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/inference_session.h"

int main(int argc, char* argv[]) {
  using namespace onnxruntime;

  SessionOptions options {};
  InferenceSession session(options);

  common::Status status = session.Load("model/mul_1.pb");

  return EXIT_SUCCESS;
}

