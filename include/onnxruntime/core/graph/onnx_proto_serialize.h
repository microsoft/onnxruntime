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
//   * bool ParseFromZeroCopyStream(ZeroCopyInputStream*)  [protobuf] /
//          ParseFromZeroCopyStream(BinaryStream*)         [onnx-light]
//   * std::string SerializeAsString() const
//   * bool SerializeToString(std::string*) const
//   * bool SerializeToArray(void*, int) const
//   * bool SerializeToOstream(std::ostream*) const
//   * bool SerializeToFileDescriptor(int) const
//   * bool SerializeToZeroCopyStream(ZeroCopyOutputStream*) [protobuf] /
//          SerializeToZeroCopyStream(BinaryWriteStream*)    [onnx-light]
//
// onnx-light 0.1.9+ also provides google::protobuf::io compat headers
// (zero_copy_stream_impl.h, coded_stream.h) so code written against the
// protobuf ZeroCopy API compiles unchanged against onnx-light.

#include "core/graph/onnx_protobuf.h"

#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <istream>
#include <ostream>
#include <string>

namespace onnxruntime {
namespace proto_io {

// Parse a proto from a raw memory buffer.
// Uses ArrayInputStream + ParseFromZeroCopyStream so that the same ZeroCopy
// path is used for both the protobuf and the onnx-light backends.
template <typename Proto>
inline bool ParseFromArray(Proto& proto, const void* data, int size) {
  google::protobuf::io::ArrayInputStream zero_copy_input(data, size);
  return proto.ParseFromZeroCopyStream(&zero_copy_input);
}

// proto.ParseFromString(data)
template <typename Proto>
inline bool ParseFromString(Proto& proto, const std::string& data) {
  return proto.ParseFromString(data);
}

// Parse a proto from a std::istream (reads the stream to EOF).
// Uses ParseFromZeroCopyStream with IstreamInputStream so that the EOF check
// catches truncated inputs (mirrors the original protobuf-based implementation).
template <typename Proto>
inline bool ParseFromIStream(Proto& proto, std::istream& stream) {
  google::protobuf::io::IstreamInputStream zero_copy_input(&stream);
  return proto.ParseFromZeroCopyStream(&zero_copy_input) && stream.eof();
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
// Uses FileOutputStream + SerializeToZeroCopyStream so that the same buffered,
// ZeroCopy path is used for both the protobuf and the onnx-light backends.
template <typename Proto>
inline bool SaveToFileDescriptor(const Proto& proto, int fd) {
  google::protobuf::io::FileOutputStream output(fd);
  return proto.SerializeToZeroCopyStream(&output) && output.Flush();
}

}  // namespace proto_io
}  // namespace onnxruntime
