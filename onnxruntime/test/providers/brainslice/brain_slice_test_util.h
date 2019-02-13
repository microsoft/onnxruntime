#pragma once
#include <stdint.h>
#include <vector>
#include "bond_struct.h"

namespace onnxruntime {
namespace test {
bond_util::BondStruct CreateRuntimeArguments(uint32_t function_id, std::vector<uint32_t>&& data);
constexpr uint32_t InitFunctionId = 24;
constexpr uint32_t ExecuteFunctionId = 25;
}  // namespace test
}  // namespace onnxruntime
