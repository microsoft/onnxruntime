// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "onnxruntime_c_api.h"
#include "utils.h"

class ExampleEpFactory;

class ExampleEp : public OrtEp, public ApiPtrs {
 public:
  ExampleEp(ExampleEpFactory& factory, const std::string& name,
            const OrtSessionOptions& session_options, const OrtLogger& logger);

  ~ExampleEp() = default;

 private:
  static const char* ORT_API_CALL GetNameImpl(const OrtEp* this_ptr) noexcept;

  ExampleEpFactory& factory_;
  std::string name_;
  const OrtSessionOptions& session_options_;
  const OrtLogger& logger_;
};
