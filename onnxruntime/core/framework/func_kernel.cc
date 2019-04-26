#include "core/framework/func_kernel.h"
#include "core/framework/allocator.h"
namespace onnxruntime {
void* allocate_helper_func(void* allocator, size_t alignment, size_t size) {
  // Here we only align the size, we expect the underline device allocator will
  // guarantee the address alignment.
  // We may update lotus' IAllocator interface to support alignment.
  size_t rounded_bytes = (alignment * ((size + alignment - 1) / alignment));
  auto* alloc = static_cast<IAllocator*>(allocator);
  return alloc->Alloc(rounded_bytes);
}

void release_helper_func(void* allocator, void* p) {
  auto* alloc = static_cast<IAllocator*>(allocator);
  return alloc->Free(p);
}

DType ORT_type_to_c_type(MLDataType type) {
  if (type == DataTypeImpl::GetType<float>())
    return DType::TFloat32;
  else if (type == DataTypeImpl::GetType<double>())
    return DType::TDouble;
  else if (type == DataTypeImpl::GetType<int32_t>())
    return DType::TInt32;
  else if (type == DataTypeImpl::GetType<bool>())
    return DType::TBool;
  else if (type == DataTypeImpl::GetType<uint8_t>())
    return DType::TUint8;
  else if (type == DataTypeImpl::GetType<int8_t>())
    return DType::TInt8;
  else if (type == DataTypeImpl::GetType<uint16_t>())
    return DType::TUint16;
  else if (type == DataTypeImpl::GetType<int16_t>())
    return DType::TInt16;
  else if (type == DataTypeImpl::GetType<uint32_t>())
    return DType::TUint32;
  else if (type == DataTypeImpl::GetType<uint64_t>())
    return DType::TUint64;
  else if (type == DataTypeImpl::GetType<int64_t>())
    return DType::TInt64;
  else
    ORT_NOT_IMPLEMENTED("Unsupport MLType to c type.");
}

}  // namespace onnxruntime
