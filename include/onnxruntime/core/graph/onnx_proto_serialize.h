// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// Serialization helpers that provide a consistent API for ONNX proto
// (de)serialization, working with both the protobuf-based upstream `onnx`
// package and the protobuf-free `onnx-light` drop-in selected with the
// onnxruntime_USE_ONNX_LIGHT build option.
//
// Both backends expose a protobuf-compatible message API:
//   * bool ParseFromArray(const void*, int)
//   * bool ParseFromString(const std::string&)
//   * bool ParseFromIstream(std::istream*)
//   * bool ParseFromFileDescriptor(int)
//   * std::string SerializeAsString() const
//   * bool SerializeToString(std::string*) const
//   * bool SerializeToArray(void*, int) const
//   * bool SerializeToOstream(std::ostream*) const
//   * bool SerializeToFileDescriptor(int) const

#include "core/graph/onnx_protobuf.h"

#include <istream>
#include <ostream>
#include <string>

namespace onnxruntime {
namespace proto_io {

// proto.ParseFromArray(data, size)
template <typename Proto>
inline bool ParseFromArray(Proto& proto, const void* data, int size) {
  return proto.ParseFromArray(data, size);
}

// proto.ParseFromString(data)
template <typename Proto>
inline bool ParseFromString(Proto& proto, const std::string& data) {
  return proto.ParseFromString(data);
}

// Parse a proto from a std::istream (reads the stream to EOF).
template <typename Proto>
inline bool ParseFromIStream(Proto& proto, std::istream& stream) {
  return proto.ParseFromIstream(&stream);
}

// Parse a proto from an OS file descriptor (reads to EOF).
template <typename Proto>
inline bool ParseFromFileDescriptor(Proto& proto, int fd) {
  return proto.ParseFromFileDescriptor(fd);
}

// proto.SerializeToString(&out)
template <typename Proto>
inline bool SerializeToString(const Proto& proto, std::string& out) {
  return proto.SerializeToString(&out);
}

// proto.SerializeAsString()
template <typename Proto>
inline std::string SerializeAsString(const Proto& proto) {
  return proto.SerializeAsString();
}

// proto.SerializeToArray(data, size)
template <typename Proto>
inline bool SerializeToArray(const Proto& proto, void* data, int size) {
  return proto.SerializeToArray(data, size);
}

// proto.SerializeToOstream(&stream)
template <typename Proto>
inline bool SerializeToOStream(const Proto& proto, std::ostream& stream) {
  return proto.SerializeToOstream(&stream);
}

// proto.SerializeToFileDescriptor(fd)  (does not take ownership of fd)
template <typename Proto>
inline bool SerializeToFileDescriptor(const Proto& proto, int fd) {
  return proto.SerializeToFileDescriptor(fd);
}

// Writes the serialized proto to an OS file descriptor without taking ownership of it.
template <typename Proto>
inline bool SaveToFileDescriptor(const Proto& proto, int fd) {
  return proto.SerializeToFileDescriptor(fd);
}

}  // namespace proto_io
}  // namespace onnxruntime
