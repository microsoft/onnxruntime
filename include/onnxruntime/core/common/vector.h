// #include <vector>

// template <typename T>
// using AttributeVector = std::vector<T>;

#ifdef USE_MIMALLOC

#include "core/common/allocator_mimalloc.h"

template <typename T>
using FastVector = std::vector<T,allocator_mimalloc<T>>;

#else

template <typename T>
using FastVector = std::vector<T>;

#endif