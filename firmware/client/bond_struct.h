#pragma once
#include <vector>
#include <memory>
#include <stdint.h>
#include "ISA_types.c.h"

namespace onnxruntime {
namespace bond_util{
struct BondStruct;

struct Value {
  enum class Type {
    Bool,
    ISA_Mem,
    UInt32,
    Struct,
    VectorUInt32,
    VectorStruct
  };

  Value(bool x)
      : type(Type::Bool), integer(x) {}

  Value(uint32_t x)
      : type(Type::UInt32), integer(x) {}

  Value(std::vector<uint32_t>&& x)
      : type(Type::VectorUInt32), vectorUInt32(x) {}

  Value(std::vector<BondStruct>&& x)
      : type(Type::VectorStruct), vectorStruct(x) {}

  Value(BondStruct&& s)
      : type(Type::Struct), struct_(std::make_shared<BondStruct>(std::forward<BondStruct>(s))) {}

  Value(ISA_Mem mem)
      : type(Type::ISA_Mem), isa_mem(mem) {}

  Type type;
  bool boolean;
  uint32_t integer;
  ISA_Mem isa_mem;
  std::vector<uint32_t> vectorUInt32;
  std::vector<BondStruct> vectorStruct;
  std::shared_ptr<BondStruct> struct_;
};

struct Tag {
  std::string name;
  std::uint16_t id;
};

struct Field {
  typedef std::vector<std::pair<std::string, std::string>> Attributes;
  
  Field(Tag t, Attributes a, Value v) : tag(t), attributes(a), value(v) {}

  Tag tag;
  Attributes attributes;
  Value value;
};

// Struct type represents a Bond struct
struct BondStruct {
  typedef std::vector<Field> Fields;
  
  BondStruct() {}

  BondStruct(Fields&& fields)
      : fields(fields){}

  Fields fields;
};
}
}  // namespace onnxruntime

