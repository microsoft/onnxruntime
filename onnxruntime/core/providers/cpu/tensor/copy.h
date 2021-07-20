#include "core/providers/common.h"
#include "core/platform/threadpool.h"
#include <vector>

namespace onnxruntime {

void CoalesceDimensions(
    std::initializer_list<std::reference_wrapper<std::vector<int64_t>>>&& tensors_strides, std::vector<int64_t>& shape);

Status DispatchStridedCopy(concurrency::ThreadPool* thread_pool,
                           Tensor& dst,
                           std::ptrdiff_t dst_offset,
                           const std::vector<int64_t> dst_strides,
                           const TensorShape& copy_shape,
                           const Tensor& src,
                           const std::vector<int64_t> src_strides);

template <typename T>
void StridedCopy(concurrency::ThreadPool* thread_pool,
                 T* dst,
                 const std::vector<int64_t>& dst_strides,
                 const TensorShape& copy_shape,
                 const T* src,
                 const std::vector<int64_t>& src_strides);

std::vector<int64_t> StridesForTensor(const Tensor& tensor);
}  // namespace onnxruntime
