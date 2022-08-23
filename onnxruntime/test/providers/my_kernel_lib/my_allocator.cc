#include "my_allocator.h"
#include <stdlib.h>

namespace my_kernel_lib {
void* my_alloc(size_t size) {
  return malloc(size);
}

void my_free(void* p) {
  free(p);
}

}  // namespace my_kernel_lib
