#include "core/common/safeint.h"
#include "core/framework/allocator.h"
#include "core/framework/tensor.h"
#include "core/framework/op_kernel.h"
#include "core/graph/onnx_protobuf.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {

namespace utils {
void* DefaultAlloc(size_t size) {
  if (size <= 0) return nullptr;
  void* p;
  const size_t alignment = 32;
#if _MSC_VER
  p = _aligned_malloc(size, alignment);
  if (p == nullptr)
    ORT_THROW_EX(std::bad_alloc);
#elif defined(_LIBCPP_SGX_CONFIG)
  p = memalign(alignment, size);
  if (p == nullptr)
    ORT_THROW_EX(std::bad_alloc);
#else
  int ret = posix_memalign(&p, alignment, size);
  if (ret != 0)
    ORT_THROW_EX(std::bad_alloc);
#endif
  return p;
}

void DefaultFree(void* p) {
#if _MSC_VER
  _aligned_free(p);
#else
  free(p);
#endif
}

}  // namespace utils

/*
// FROM allocator.cc
//
void* CPUAllocator::Alloc(size_t size) {
  if (size <= 0) return nullptr;
  void* p;
  const size_t alignment = 32;
#if defined(_LIBCPP_SGX_CONFIG)
  p = memalign(alignment, size);
  if (p == nullptr)
    ORT_THROW_EX(std::bad_alloc);
#elif defined(_MSC_VER)
  p = _aligned_malloc(size, alignment);
  if (p == nullptr)
    ORT_THROW_EX(std::bad_alloc);
#else
  int ret = posix_memalign(&p, alignment, size);
  if (ret != 0)
    ORT_THROW_EX(std::bad_alloc);
#endif
  return p;
}

void CPUAllocator::Free(void* p) {
  free(p);
}

bool IAllocator::CalcMemSizeForArrayWithAlignment(size_t nmemb, size_t size, size_t alignment, size_t* out) noexcept {
  bool ok = true;

  ORT_TRY {
    SafeInt<size_t> alloc_size(size);
    if (alignment == 0) {
      *out = alloc_size * nmemb;
    } else {
      size_t alignment_mask = alignment - 1;
      *out = (alloc_size * nmemb + alignment_mask) & ~static_cast<size_t>(alignment_mask);
    }
  }
  ORT_CATCH(const OnnxRuntimeException& ex) {
    // overflow in calculating the size thrown by SafeInt.
    ORT_HANDLE_EXCEPTION([&]() {
      LOGS_DEFAULT(ERROR) << ex.what();
      ok = false;
    });
  }
  return ok;
}

OrtValue* OpKernelContext::OutputMLValue(int index, const TensorShape& shape, size_t nnz) {
  if (index < 0 || index >= OutputCount())
    return nullptr;

  OrtValue* p_ort_value = &values_[GetOutputArgIndex(index)];
  MLDataType element_type = types_[GetOutputArgIndex(index)];
  if (p_ort_value->IsAllocated()) {
    if (p_ort_value->IsTensor()) {
      const Tensor& tensor = p_ort_value->Get<Tensor>();
      ORT_ENFORCE(tensor.Shape() == shape,
                  "OrtValue shape verification failed. Current shape:", tensor.Shape(),
                  " Requested shape:", shape.ToString());
    }
  } else {
    size_t size;
    int64_t len = shape.Size();
    ORT_ENFORCE(len >= 0, "Tensor shape cannot contain any negative value");
    ORT_ENFORCE(static_cast<uint64_t>(len) < std::numeric_limits<size_t>::max(), "Tensor shape is too large");
    ORT_ENFORCE(IAllocator::CalcMemSizeForArrayWithAlignment<64>(static_cast<size_t>(len), element_type->Size(), &size), "size overflow");

    AllocatorPtr alloc = kernel_->Info().GetAllocator(0, OrtMemTypeDefault);
    std::unique_ptr<Tensor> p_tensor = onnxruntime::make_unique<Tensor>(element_type, shape, alloc);
    {
      auto ml_tensor = DataTypeImpl::GetType<Tensor>();
      p_ort_value->Init(p_tensor.release(), ml_tensor, ml_tensor->GetDeleteFunc());
    }
  }

  return p_ort_value;
}
*/

}  // namespace onnxruntime

namespace ONNX_NAMESPACE {

std::ostream& operator<<(std::ostream& out, const TensorShapeProto& /*shape_proto*/) {
  std::string result;
  return (out << result);
}

std::ostream& operator<<(std::ostream& out, const TensorProto& /*tensor_proto*/) {
  std::string result;
  return (out << result);
}

}  // namespace ONNX_NAMESPACE
