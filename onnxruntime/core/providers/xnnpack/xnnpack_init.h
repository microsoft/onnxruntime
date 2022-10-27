#include "core/framework/allocator.h"

struct xnn_allocator;

namespace onnxruntime {
namespace xnnpack {

std::pair<AllocatorPtr, xnn_allocator*> GetOrCreateAllocator();

}  // namespace xnnpack
}  // namespace onnxruntime
