// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include "core/common/inlined_containers.h"

namespace onnxruntime {
namespace cloud {

struct Data {
  char* content{};
  size_t size_in_byte{};
};

using EndPointConfig = onnxruntime::InlinedHashMap<std::string, std::string>;

enum EndPointType {
  Rest,
  //Soap
  Unknown,
};

class EndPointInvoker {
 public:
  EndPointInvoker(){};
  virtual ~EndPointInvoker(){};
  virtual Data Send(Data data) const; // Send is stateless
  static EndPointType MapType(const std::string&);
  static std::unique_ptr<EndPointInvoker> CreateInvoker(EndPointType type, const EndPointConfig& config);
};

class RestInvoker : public EndPointInvoker {
 public:
  RestInvoker(const EndPointConfig& config);
  Data Send(Data) const override;

 private:
  const EndPointConfig& config_;
};

//class SoapInvoker : public EndPointInvoker {
// public:
//
//};

}  // namespace cloud
}  // namespace onnxruntime