#include <atomic>
#include "core/framework/allocator.h"
#include "core/platform/ort_mutex.h"
#include "xnnpack.h"

namespace onnxruntime {
namespace xnnpack {

class XnnpackInitWrapper {
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(XnnpackInitWrapper);

 public:
  // thread safe
  static XnnpackInitWrapper& GetInstance() {
    static XnnpackInitWrapper instance;
    return instance;
  }

  void InitXnnpackWithAllocatorAndAddRef(AllocatorPtr allocator);
  AllocatorPtr GetOrCreateAllocator();

  void increase_ref() {
    ++allocator_ref_count_;
  }

  void release_ref() {
    --allocator_ref_count_;
    if (allocator_ref_count_ == 0) {
      ort_allocator_.reset();
    }
  }

 private:
  XnnpackInitWrapper() = default;
  ~XnnpackInitWrapper() { xnn_deinitialize(); }

 private:
  std::atomic_int allocator_ref_count_{0};
  AllocatorPtr ort_allocator_;
  xnn_allocator xnn_allocator_wrapper_;
  OrtMutex mutex;
};
}  // namespace xnnpack
}  // namespace onnxruntime
