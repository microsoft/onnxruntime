// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Protobuf-free replacement for the protoc-generated tml.pb.h, used when
// ONNX Runtime is built against onnx-light (ORT_USE_ONNX_LIGHT) with neither
// the `onnx` package nor protobuf available.
//
// It hand-implements the subset of onnxruntime/test/proto/tml.proto that the
// test tooling (test/onnx/TestCase.cc and pb_helper) actually consumes:
//   - the eight Map<K,V> message types,
//   - VectorMapStringToFloat / VectorMapInt64ToFloat,
//   - TraditionalMLData (the oneof container plus name / debug_info / tensor).
// Only *parsing* is implemented: tml test-data files are produced offline by
// Python (which still uses the real onnx/protobuf), and the C++ side only ever
// reads them.
//
// The wire format is standard protobuf, decoded through onnx-light's
// utils::BinaryStream primitives (next_field / next_uint64 / next_float /
// next_double / next_string / LimitToNext / Restore). protobuf `map<K,V>` fields
// are wire-identical to `repeated Entry { key = 1; value = 2; }`, which is how
// they are decoded here.
#pragma once

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>

#include <onnx/onnx_pb.h>

#include "core/common/inlined_containers.h"

namespace onnxruntime {
namespace proto {

namespace ol = ONNX_LIGHT_NAMESPACE;

namespace tml_detail {

// protobuf wire types used by tml.
enum WireType : uint64_t {
  kWireVarint = 0,
  kWire64Bit = 1,
  kWireLengthDelimited = 2,
  kWire32Bit = 5,
};

// Advances past a single field whose value is not consumed by the caller.
inline void SkipField(ol::utils::BinaryStream& s, uint64_t wire_type) {
  switch (wire_type) {
    case kWireVarint:
      s.next_uint64();
      break;
    case kWire64Bit:
      s.skip_bytes(8);
      break;
    case kWireLengthDelimited: {
      const uint64_t len = s.next_uint64();
      s.skip_bytes(static_cast<ol::utils::offset_t>(len));
      break;
    }
    case kWire32Bit:
      s.skip_bytes(4);
      break;
    default:
      throw std::runtime_error("tml_onnx_light: unsupported wire type while parsing.");
  }
}

// Reads a single scalar of type T from the stream (the field tag has already
// been consumed). tml uses plain protobuf `int64` (not sint64), so integers are
// decoded as plain varints, not ZigZag.
template <typename T>
T ReadScalar(ol::utils::BinaryStream& s);

template <>
inline std::string ReadScalar<std::string>(ol::utils::BinaryStream& s) {
  const ol::utils::RefString rs = s.next_string();
  return std::string(rs.sv());
}

template <>
inline int64_t ReadScalar<int64_t>(ol::utils::BinaryStream& s) {
  return static_cast<int64_t>(s.next_uint64());
}

template <>
inline float ReadScalar<float>(ol::utils::BinaryStream& s) {
  return s.next_float();
}

template <>
inline double ReadScalar<double>(ol::utils::BinaryStream& s) {
  return s.next_double();
}

// Reads one length-delimited entry of a protobuf map (key = field 1,
// value = field 2) into *out*. The stream must already be limited to the entry.
template <typename K, typename V>
inline void ParseMapEntry(ol::utils::BinaryStream& s, std::map<K, V>& out) {
  K key{};
  V value{};
  while (s.NotEnd()) {
    const ol::utils::FieldNumber f = s.next_field();
    if (f.field_number == 1) {
      key = ReadScalar<K>(s);
    } else if (f.field_number == 2) {
      value = ReadScalar<V>(s);
    } else {
      SkipField(s, f.wire_type);
    }
  }
  out[key] = value;
}

}  // namespace tml_detail

// Message wrapping a protobuf `map<K, V> v = 1;`.
template <typename K, typename V>
class MapMessage {
 public:
  using MapType = std::map<K, V>;

  const MapType& v() const { return v_; }
  MapType& v() { return v_; }
  void Clear() { v_.clear(); }

  void ParseFromStream(ol::utils::BinaryStream& s, ol::ParseOptions& /*opts*/) {
    while (s.NotEnd()) {
      const ol::utils::FieldNumber f = s.next_field();
      if (f.field_number == 1 && f.wire_type == tml_detail::kWireLengthDelimited) {
        const uint64_t len = s.next_uint64();
        s.LimitToNext(len);
        tml_detail::ParseMapEntry<K, V>(s, v_);
        s.Restore();
      } else {
        tml_detail::SkipField(s, f.wire_type);
      }
    }
  }

 private:
  MapType v_;
};

using MapStringToString = MapMessage<std::string, std::string>;
using MapStringToInt64 = MapMessage<std::string, int64_t>;
using MapStringToFloat = MapMessage<std::string, float>;
using MapStringToDouble = MapMessage<std::string, double>;
using MapInt64ToString = MapMessage<int64_t, std::string>;
using MapInt64ToInt64 = MapMessage<int64_t, int64_t>;
using MapInt64ToFloat = MapMessage<int64_t, float>;
using MapInt64ToDouble = MapMessage<int64_t, double>;

// Message wrapping a protobuf `repeated MapMessage v = 1;`.
template <typename MapMsg>
class VectorMapMessage {
 public:
  using RepeatedType = onnxruntime::InlinedVector<MapMsg>;

  const RepeatedType& v() const { return v_; }
  RepeatedType& v() { return v_; }
  void Clear() { v_.clear(); }

  void ParseFromStream(ol::utils::BinaryStream& s, ol::ParseOptions& opts) {
    while (s.NotEnd()) {
      const ol::utils::FieldNumber f = s.next_field();
      if (f.field_number == 1 && f.wire_type == tml_detail::kWireLengthDelimited) {
        const uint64_t len = s.next_uint64();
        s.LimitToNext(len);
        v_.emplace_back();
        v_.back().ParseFromStream(s, opts);
        s.Restore();
      } else {
        tml_detail::SkipField(s, f.wire_type);
      }
    }
  }

 private:
  RepeatedType v_;
};

using VectorMapStringToFloat = VectorMapMessage<MapStringToFloat>;
using VectorMapInt64ToFloat = VectorMapMessage<MapInt64ToFloat>;

// The oneof container. Mirrors the generated TraditionalMLData surface used by
// TestCase.cc: values_case() + the kXxx case constants, the message accessors,
// name(), debug_info() and Clear().
class TraditionalMLData {
 public:
  enum ValuesCase {
    VALUES_NOT_SET = 0,
    kMapStringToString = 1,
    kMapStringToInt64 = 2,
    kMapStringToFloat = 3,
    kMapStringToDouble = 4,
    kMapInt64ToString = 5,
    kMapInt64ToInt64 = 6,
    kMapInt64ToFloat = 7,
    kMapInt64ToDouble = 8,
    kVectorString = 9,
    kVectorFloat = 10,
    kVectorInt64 = 11,
    kVectorDouble = 12,
    kVectorMapStringToFloat = 13,
    kVectorMapInt64ToFloat = 14,
    kTensor = 16,
  };

  ValuesCase values_case() const { return values_case_; }

  const MapStringToString& map_string_to_string() const { return map_string_to_string_; }
  const MapStringToInt64& map_string_to_int64() const { return map_string_to_int64_; }
  const MapStringToFloat& map_string_to_float() const { return map_string_to_float_; }
  const MapStringToDouble& map_string_to_double() const { return map_string_to_double_; }
  const MapInt64ToString& map_int64_to_string() const { return map_int64_to_string_; }
  const MapInt64ToInt64& map_int64_to_int64() const { return map_int64_to_int64_; }
  const MapInt64ToFloat& map_int64_to_float() const { return map_int64_to_float_; }
  const MapInt64ToDouble& map_int64_to_double() const { return map_int64_to_double_; }
  const VectorMapStringToFloat& vector_map_string_to_float() const {
    return vector_map_string_to_float_;
  }
  const VectorMapInt64ToFloat& vector_map_int64_to_float() const {
    return vector_map_int64_to_float_;
  }
  const ONNX_NAMESPACE::TensorProto& tensor() const { return tensor_; }

  const std::string& name() const { return name_; }
  const std::string& debug_info() const { return debug_info_; }

  void Clear() {
    values_case_ = VALUES_NOT_SET;
    map_string_to_string_.Clear();
    map_string_to_int64_.Clear();
    map_string_to_float_.Clear();
    map_string_to_double_.Clear();
    map_int64_to_string_.Clear();
    map_int64_to_int64_.Clear();
    map_int64_to_float_.Clear();
    map_int64_to_double_.Clear();
    vector_map_string_to_float_.Clear();
    vector_map_int64_to_float_.Clear();
    tensor_.Clear();
    name_.clear();
    debug_info_.clear();
  }

  void ParseFromStream(ol::utils::BinaryStream& s, ol::ParseOptions& opts) {
    while (s.NotEnd()) {
      const ol::utils::FieldNumber f = s.next_field();
      switch (f.field_number) {
        case kMapStringToString:
          ParseSub(s, opts, map_string_to_string_, kMapStringToString);
          break;
        case kMapStringToInt64:
          ParseSub(s, opts, map_string_to_int64_, kMapStringToInt64);
          break;
        case kMapStringToFloat:
          ParseSub(s, opts, map_string_to_float_, kMapStringToFloat);
          break;
        case kMapStringToDouble:
          ParseSub(s, opts, map_string_to_double_, kMapStringToDouble);
          break;
        case kMapInt64ToString:
          ParseSub(s, opts, map_int64_to_string_, kMapInt64ToString);
          break;
        case kMapInt64ToInt64:
          ParseSub(s, opts, map_int64_to_int64_, kMapInt64ToInt64);
          break;
        case kMapInt64ToFloat:
          ParseSub(s, opts, map_int64_to_float_, kMapInt64ToFloat);
          break;
        case kMapInt64ToDouble:
          ParseSub(s, opts, map_int64_to_double_, kMapInt64ToDouble);
          break;
        case kVectorMapStringToFloat:
          ParseSub(s, opts, vector_map_string_to_float_, kVectorMapStringToFloat);
          break;
        case kVectorMapInt64ToFloat:
          ParseSub(s, opts, vector_map_int64_to_float_, kVectorMapInt64ToFloat);
          break;
        case kVectorString:
        case kVectorFloat:
        case kVectorInt64:
        case kVectorDouble:
          // Plain-vector payloads are not consumed by the test tooling; record
          // the active case and skip the message body.
          values_case_ = static_cast<ValuesCase>(f.field_number);
          tml_detail::SkipField(s, f.wire_type);
          break;
        case kTensor: {
          if (f.wire_type != tml_detail::kWireLengthDelimited) {
            tml_detail::SkipField(s, f.wire_type);
            break;
          }
          const uint64_t len = s.next_uint64();
          s.LimitToNext(len);
          tensor_.ParseFromStream(s, opts);
          s.Restore();
          values_case_ = kTensor;
          break;
        }
        case 15:  // name
          name_ = tml_detail::ReadScalar<std::string>(s);
          break;
        case 17:  // debug_info
          debug_info_ = tml_detail::ReadScalar<std::string>(s);
          break;
        default:
          tml_detail::SkipField(s, f.wire_type);
          break;
      }
    }
  }

 private:
  template <typename Msg>
  void ParseSub(ol::utils::BinaryStream& s, ol::ParseOptions& opts, Msg& msg, ValuesCase c) {
    const uint64_t len = s.next_uint64();
    s.LimitToNext(len);
    msg.ParseFromStream(s, opts);
    s.Restore();
    values_case_ = c;
  }

  ValuesCase values_case_ = VALUES_NOT_SET;
  MapStringToString map_string_to_string_;
  MapStringToInt64 map_string_to_int64_;
  MapStringToFloat map_string_to_float_;
  MapStringToDouble map_string_to_double_;
  MapInt64ToString map_int64_to_string_;
  MapInt64ToInt64 map_int64_to_int64_;
  MapInt64ToFloat map_int64_to_float_;
  MapInt64ToDouble map_int64_to_double_;
  VectorMapStringToFloat vector_map_string_to_float_;
  VectorMapInt64ToFloat vector_map_int64_to_float_;
  ONNX_NAMESPACE::TensorProto tensor_;
  std::string name_;
  std::string debug_info_;
};

}  // namespace proto
}  // namespace onnxruntime
