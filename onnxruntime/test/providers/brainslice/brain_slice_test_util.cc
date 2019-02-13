#include "test/providers/brainslice/brain_slice_test_util.h"

namespace onnxruntime {
namespace test {
bond_util::BondStruct CreateRuntimeArguments(uint32_t function_id, std::vector<uint32_t>&& data) {
  return bond_util::BondStruct({
      {{"functionId", 0}, {}, bond_util::Value(function_id)},
      {{"data", 0}, {}, bond_util::Value(std::move(data))},
  });
}
}  // namespace test
}  // namespace onnxruntime
