#pragma once
#include <vector>
#include <stdint.h>

namespace my_kernel_lib {

void* my_alloc(size_t size);
void my_free(void* p);
}  // namespace my_kernel_lib
