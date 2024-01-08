#pragma once

#include <cstddef>

namespace onnx_c_ops {

void *AllocatorDefaultAlloc(std::size_t size);
void AllocatorDefaultFree(void *p);

} // namespace onnx_c_ops
