#pragma once
#include <bond/core/bond.h>
#include <bond/stream/stdio_output_stream.h>
#include <bond/protocol/simple_json_writer.h>
#include <stdint.h>
#include "bond_struct.h"

namespace onnxruntime {
namespace bond_util{
// Implement Bond parser interface for objects of type Struct
class StructParser {
 public:
  StructParser(const BondStruct& struct_)
      : struct_(struct_) {}

  template <typename Transform>
  bool Apply(const Transform& transform) {
    bond::Metadata metadata;

    transform.Begin(metadata);
    ReadFields(transform);
    transform.End();

    return true;
  }

 protected:
  template <typename Transform>
  void ReadFields(const Transform& transform) {
    for (auto const& x : struct_.fields) {
      bond::Metadata metadata;
      metadata.name = x.tag.name;
      metadata.modifier = bond::Modifier::Optional;

      for (auto& attribute : x.attributes)
        metadata.attributes.insert(attribute);

      switch (x.value.type) {
        case Value::Type::Bool:
          transform.Field(x.tag.id, metadata, x.value.boolean);
          break;

        case Value::Type::ISA_Mem:
          transform.Field(x.tag.id, metadata, x.value.isa_mem);
          break;

        case Value::Type::UInt32:
          transform.Field(x.tag.id, metadata, x.value.integer);
          break;

        case Value::Type::VectorUInt32:
          transform.Field(x.tag.id, metadata, x.value.vectorUInt32);
          break;

        case Value::Type::VectorStruct:
          transform.Field(x.tag.id, metadata, x.value.vectorStruct);
          break;

        case Value::Type::Struct:
          transform.Field(x.tag.id, metadata, *x.value.struct_);
          break;
      }
    }
  }

  const BondStruct& struct_;
};
}
}  // namespace BrainSlice

namespace bond {
// Tell Bond that our type Struct should be treated a Bond struct...
template <>
struct
    is_bond_type<onnxruntime::bond_util::BondStruct>
    : std::true_type {};

// ...and how to parse objects of the Struct type
template <typename Protocols, typename Transform>
bool Apply(const Transform& transform, const onnxruntime::bond_util::BondStruct& value) {
  return onnxruntime::bond_util::StructParser(value).Apply(transform);
}
}  // namespace bond
