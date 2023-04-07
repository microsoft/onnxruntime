#include "core/framework/allocator.h"

struct xnn_allocator;

namespace onnxruntime {
namespace xnnpack {

std::pair<AllocatorPtr&, xnn_allocator*> GetStoredAllocator();

}  // namespace xnnpack
}  // namespace onnxruntime
