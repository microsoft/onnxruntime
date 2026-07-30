// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// Both the protobuf-based upstream `onnx` package and the protobuf-free
// `onnx-light` drop-in (selected with onnxruntime_USE_ONNX_LIGHT) expose a
// protobuf-compatible message API directly on the generated proto classes:
//
//   * bool ParseFromArray(const void*, int)
//   * bool ParseFromString(const std::string&)
//   * bool ParseFromIstream(std::istream*)
//   * bool ParseFromFileDescriptor(int)
//   * bool ParseFromZeroCopyStream(ZeroCopyInputStream*)
//   * std::string SerializeAsString() const
//   * bool SerializeToString(std::string*) const
//   * bool SerializeToArray(void*, int) const
//   * bool SerializeToOstream(std::ostream*) const
//   * bool SerializeToZeroCopyStream(ZeroCopyOutputStream*)
//
// onnx-light 0.1.9+ also provides google::protobuf::io compat headers
// (zero_copy_stream_impl.h, coded_stream.h) so code written against the
// protobuf ZeroCopy API compiles unchanged against onnx-light.
//
// Call the proto methods directly. For serialization to / parsing from an OS
// file descriptor using the ZeroCopy path, use FileOutputStream /
// IstreamInputStream from <google/protobuf/io/zero_copy_stream_impl.h>.

#include "core/graph/onnx_protobuf.h"

#include <google/protobuf/io/zero_copy_stream_impl.h>
