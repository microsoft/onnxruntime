#pragma once
// Protocol Buffers - Google's data interchange format
// Copyright 2008 Google Inc.  All rights reserved.
// https://developers.google.com/protocol-buffers/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "onnxruntime_config.h"
#ifdef HAS_SHORTEN_64_TO_32
#pragma GCC diagnostic ignored "-Wshorten-64-to-32"
#endif
#endif
#if defined(ORT_USE_ONNX_LIGHT)
#include <onnx_proto/google_protobuf_compat.h>
#include <onnx_lib/common/onnx_pb.h>
#include <onnx_lib/onnx-data.pb.h>
#include <cstdint>
#include <onnx_proto/tml.h>
namespace onnxruntime {
namespace proto {
using ONNX_LIGHT_NAMESPACE::proto::MapInt64ToDouble;
using ONNX_LIGHT_NAMESPACE::proto::MapInt64ToFloat;
using ONNX_LIGHT_NAMESPACE::proto::MapInt64ToInt64;
using ONNX_LIGHT_NAMESPACE::proto::MapInt64ToString;
using ONNX_LIGHT_NAMESPACE::proto::MapStringToDouble;
using ONNX_LIGHT_NAMESPACE::proto::MapStringToFloat;
using ONNX_LIGHT_NAMESPACE::proto::MapStringToInt64;
using ONNX_LIGHT_NAMESPACE::proto::MapStringToString;
using ONNX_LIGHT_NAMESPACE::proto::TraditionalMLData;
using ONNX_LIGHT_NAMESPACE::proto::VectorMapInt64ToFloat;
using ONNX_LIGHT_NAMESPACE::proto::VectorMapStringToFloat;
}  // namespace proto
}  // namespace onnxruntime
#else
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <onnx/onnx_pb.h>
#include <onnx/onnx-data.pb.h>
#include <google/protobuf/message_lite.h>
#include "tml.pb.h"
#endif
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
namespace onnxruntime {
template <typename Msg>
bool ParseDelimitedFromCodedStream(Msg* message,
                                   google::protobuf::io::CodedInputStream* input,
                                   bool* clean_eof) {
  if (clean_eof != nullptr) *clean_eof = false;
  int start = input->CurrentPosition();

  uint32_t size;
  if (!input->ReadVarint32(&size)) {
    if (clean_eof != nullptr) *clean_eof = input->CurrentPosition() == start;
    return false;
  }

  std::string buf(static_cast<size_t>(size), '\0');
  if (!input->ReadRaw(buf.data(), static_cast<int>(size))) return false;
  return message->ParseFromArray(buf.data(), static_cast<int>(size));
}
}  // namespace onnxruntime
