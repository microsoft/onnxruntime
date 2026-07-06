// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// Serialization helpers that abstract the difference between the protobuf-based
// upstream `onnx` package and the protobuf-free `onnx-light` drop-in that is
// selected with the onnxruntime_USE_ONNX_LIGHT build option.
//
// When ORT is built against onnx-light the ONNX message classes (ModelProto,
// TensorProto, ...) do NOT expose the protobuf message API
// (ParseFromArray/SerializeToArray/ParseFromZeroCopyStream/...).
// They provide onnx-light's own API instead:
//   * bool   ParseFromString(const std::string&)
//   * bool   ParseFromIstream(std::istream*)
//   * bool   SerializeToString(std::string&) const
//   * size_t ByteSizeLong() const  (native; computed from SerializeSize())
// These helpers route every (de)serialization operation used by onnxruntime to
// the correct backend. When onnx-light is not used they forward verbatim to the
// protobuf API, preserving the existing behavior byte for byte.
//
// See the onnx-light documentation page "Replacing onnxruntime's protobuf usage"
// for the full mapping of protobuf message/stream API to onnx-light API.

#include "core/graph/onnx_protobuf.h"

#include <cstddef>
#include <cstring>
#include <istream>
#include <ostream>
#include <string>

#if !defined(ORT_USE_ONNX_LIGHT)
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#else
#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif
#endif

namespace onnxruntime {
namespace proto_io {

#if defined(ORT_USE_ONNX_LIGHT)
namespace detail {
// Writes the whole buffer to an OS file descriptor. Returns false on I/O error.
inline bool WriteAllToFd(int fd, const std::string& buffer) {
  size_t off = 0;
  while (off < buffer.size()) {
    const size_t remaining = buffer.size() - off;
    const int chunk = static_cast<int>(remaining < (size_t{1} << 20) ? remaining : (size_t{1} << 20));
#if defined(_WIN32)
    const int n = _write(fd, buffer.data() + off, static_cast<unsigned int>(chunk));
#else
    const auto n = ::write(fd, buffer.data() + off, static_cast<size_t>(chunk));
#endif
    if (n < 0) {
      return false;
    }
    off += static_cast<size_t>(n);
  }
  return true;
}

// Reads a whole OS file descriptor into a string. Returns false on I/O error.
inline bool ReadAllFromFd(int fd, std::string& buffer) {
  char chunk[1 << 16];
  for (;;) {
#if defined(_WIN32)
    const int n = _read(fd, chunk, static_cast<unsigned int>(sizeof(chunk)));
#else
    const auto n = ::read(fd, chunk, sizeof(chunk));
#endif
    if (n < 0) {
      return false;
    }
    if (n == 0) {
      break;
    }
    buffer.append(chunk, static_cast<size_t>(n));
  }
  return true;
}
}  // namespace detail
#endif  // ORT_USE_ONNX_LIGHT

// proto.ParseFromArray(data, size)
template <typename Proto>
inline bool ParseFromArray(Proto& proto, const void* data, int size) {
#if defined(ORT_USE_ONNX_LIGHT)
  proto.ParseFromString(std::string(reinterpret_cast<const char*>(data), static_cast<size_t>(size)));
  return true;
#else
  return proto.ParseFromArray(data, size);
#endif
}

// proto.ParseFromString(data)  (protobuf returns bool, onnx-light returns void)
template <typename Proto>
inline bool ParseFromString(Proto& proto, const std::string& data) {
#if defined(ORT_USE_ONNX_LIGHT)
  proto.ParseFromString(data);
  return true;
#else
  return proto.ParseFromString(data);
#endif
}

// Parse a proto from a std::istream (reads the stream to EOF).
template <typename Proto>
inline bool ParseFromIStream(Proto& proto, std::istream& stream) {
#if defined(ORT_USE_ONNX_LIGHT)
  return proto.ParseFromIstream(&stream);
#else
  google::protobuf::io::IstreamInputStream zero_copy_input(&stream);
  return proto.ParseFromZeroCopyStream(&zero_copy_input) && stream.eof();
#endif
}

// Parse a proto from an OS file descriptor (reads to EOF).
template <typename Proto>
inline bool ParseFromFileDescriptor(Proto& proto, int fd) {
#if defined(ORT_USE_ONNX_LIGHT)
  std::string buffer;
  if (!detail::ReadAllFromFd(fd, buffer)) {
    return false;
  }
  proto.ParseFromString(buffer);
  return true;
#else
  google::protobuf::io::FileInputStream fs(fd);
  return proto.ParseFromZeroCopyStream(&fs) && fs.GetErrno() == 0;
#endif
}

// proto.SerializeToString(&out)  (protobuf returns bool, onnx-light returns void)
template <typename Proto>
inline bool SerializeToString(const Proto& proto, std::string& out) {
#if defined(ORT_USE_ONNX_LIGHT)
  proto.SerializeToString(out);
  return true;
#else
  return proto.SerializeToString(&out);
#endif
}

// proto.SerializeAsString()
template <typename Proto>
inline std::string SerializeAsString(const Proto& proto) {
#if defined(ORT_USE_ONNX_LIGHT)
  std::string out;
  proto.SerializeToString(out);
  return out;
#else
  return proto.SerializeAsString();
#endif
}

// proto.ByteSizeLong()  (serialized size in bytes)
template <typename Proto>
inline size_t ByteSize(const Proto& proto) {
  // onnx-light's ByteSizeLong() computes the size via SerializeSize() without
  // performing a real serialization, matching protobuf's ByteSizeLong().
  return proto.ByteSizeLong();
}

// proto.SerializeToArray(data, size)
template <typename Proto>
inline bool SerializeToArray(const Proto& proto, void* data, int size) {
#if defined(ORT_USE_ONNX_LIGHT)
  std::string out;
  proto.SerializeToString(out);
  if (static_cast<int>(out.size()) > size) {
    return false;
  }
  std::memcpy(data, out.data(), out.size());
  return true;
#else
  return proto.SerializeToArray(data, size);
#endif
}

// proto.SerializeToOstream(&stream)
template <typename Proto>
inline bool SerializeToOStream(const Proto& proto, std::ostream& stream) {
#if defined(ORT_USE_ONNX_LIGHT)
  std::string out;
  proto.SerializeToString(out);
  stream.write(out.data(), static_cast<std::streamsize>(out.size()));
  return stream.good();
#else
  return proto.SerializeToOstream(&stream);
#endif
}

// proto.SerializeToFileDescriptor(fd)  (does not take ownership of fd)
template <typename Proto>
inline bool SerializeToFileDescriptor(const Proto& proto, int fd) {
#if defined(ORT_USE_ONNX_LIGHT)
  std::string out;
  proto.SerializeToString(out);
  return detail::WriteAllToFd(fd, out);
#else
  return proto.SerializeToFileDescriptor(fd);
#endif
}

// Equivalent of `FileOutputStream output(fd); proto.SerializeToZeroCopyStream(&output) && output.Flush();`
// Writes the serialized proto to an OS file descriptor without taking ownership of it.
template <typename Proto>
inline bool SaveToFileDescriptor(const Proto& proto, int fd) {
#if defined(ORT_USE_ONNX_LIGHT)
  std::string out;
  proto.SerializeToString(out);
  return detail::WriteAllToFd(fd, out);
#else
  google::protobuf::io::FileOutputStream output(fd);
  return proto.SerializeToZeroCopyStream(&output) && output.Flush();
#endif
}

}  // namespace proto_io
}  // namespace onnxruntime
