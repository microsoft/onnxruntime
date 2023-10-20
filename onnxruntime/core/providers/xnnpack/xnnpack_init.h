#include "core/framework/allocator.h"

struct xnn_allocator;

namespace onnxruntime {
namespace xnnpack {

// TODO: This is used for workspace allocations.
// It could be refined if needed to match what happens in xnnpack/src/common.h as the alignment could be reduced to
// 16 or 32 depending on the platform. However the workspace is a single allocation for a kernel during Compute,
// that is used in a subset of kernels, so saving the bytes is not expected to make a meaningful difference to
// justify the effort of refining it.
#define XNN_ALLOCATION_ALIGNMENT 64

std::pair<AllocatorPtr&, xnn_allocator*> GetStoredAllocator();

}  // namespace xnnpack
}  // namespace onnxruntime
