//-----------------------------------------------------------------------------
// MurmurHash3 was written by Austin Appleby, and is placed in the public
// domain. The author hereby disclaims copyright to this source code.

//scikit-learn is a Python module for machine learning built on top of SciPy and
//distributed under the 3-Clause BSD license. See https://github.com/scikit-learn/scikit-learn.
//This material is licensed under the BSD License (see https://github.com/scikit-learn/scikit-learn/blob/master/COPYING);
/* Modifications Copyright (c) Microsoft. */

#include "contrib_ops/cpu/murmur_hash3.h"

// Platform-specific functions and macros

// Microsoft Visual Studio

#if defined(_MSC_VER)

#define FORCE_INLINE __forceinline

#include <stdlib.h>

#define ROTL32(x, y) _rotl(x, y)
#define ROTL64(x, y) _rotl64(x, y)

#define BIG_CONSTANT(x) (x)

// Other compilers

#else  // defined(_MSC_VER)

#if defined(GNUC) && ((GNUC > 4) || (GNUC == 4 && GNUC_MINOR >= 4))

// gcc version >= 4.4 4.1 = RHEL 5, 4.4 = RHEL 6.
// Don't inline for RHEL 5 gcc which is 4.1
#define FORCE_INLINE attribute((always_inline))

#else

#define FORCE_INLINE

#endif

inline uint32_t rotl32(uint32_t x, int8_t r) {
  return (x << r) | (x >> (32 - r));
}

inline uint64_t rotl64(uint64_t x, int8_t r) {
  return (x << r) | (x >> (64 - r));
}

#define ROTL32(x, y) rotl32(x, y)
#define ROTL64(x, y) rotl64(x, y)

#define BIG_CONSTANT(x) (x##LLU)

#endif  // !defined(_MSC_VER)

//-----------------------------------------------------------------------------
// If your platform needs to do endian-swapping or can only
// handle aligned reads, do the conversion here

FORCE_INLINE uint32_t getblock(const uint32_t* p, int i) {
  return p[i];
}

FORCE_INLINE uint64_t getblock(const uint64_t* p, int i) {
  return p[i];
}

//-----------------------------------------------------------------------------
// Finalization mix - force all bits of a hash block to avalanche

FORCE_INLINE uint32_t fmix(uint32_t h) {
  h ^= h >> 16;
  h *= 0x85ebca6b;
  h ^= h >> 13;
  h *= 0xc2b2ae35;
  h ^= h >> 16;

  return h;
}

//----------

FORCE_INLINE uint64_t fmix(uint64_t k) {
  k ^= k >> 33;
  k *= BIG_CONSTANT(0xff51afd7ed558ccd);
  k ^= k >> 33;
  k *= BIG_CONSTANT(0xc4ceb9fe1a85ec53);
  k ^= k >> 33;

  return k;
}

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    MurmurHash3,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(),
                                                      DataTypeImpl::GetTensorType<uint32_t>(),
                                                      DataTypeImpl::GetTensorType<std::string>()})
        .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(),
                                                      DataTypeImpl::GetTensorType<uint32_t>()}),
    MurmurHash3);

void MurmurHash3::MurmurHash3_x86_32(const void* key, int len, uint32_t seed, void* out) const {
  const uint8_t* data = reinterpret_cast<const uint8_t*>(key);
  const int nblocks = len / 4;
  uint32_t h1 = seed;
  const uint32_t c1 = 0xcc9e2d51;
  const uint32_t c2 = 0x1b873593;

  //----------
  // body
  const uint32_t* blocks = reinterpret_cast<const uint32_t*>(data + static_cast<int64_t>(nblocks) * 4);

  for (int i = -nblocks; i; i++) {
    uint32_t k1 = getblock(blocks, i);

    k1 *= c1;
    k1 = ROTL32(k1, 15);
    k1 *= c2;

    h1 ^= k1;
    h1 = ROTL32(h1, 13);
    h1 = h1 * 5 + 0xe6546b64;
  }

  //----------
  // tail
  const uint8_t* tail = reinterpret_cast<const uint8_t*>(data + static_cast<int64_t>(nblocks) * 4);

  uint32_t k1 = 0;

  switch (len & 3) {
    case 3:
      k1 ^= tail[2] << 16;  // Fallthrough.
    case 2:
      k1 ^= tail[1] << 8;  // Fallthrough.
    case 1:
      k1 ^= tail[0];
      k1 *= c1;
      k1 = ROTL32(k1, 15);
      k1 *= c2;
      h1 ^= k1;
  };

  //----------
  // finalization
  h1 ^= len;

  h1 = fmix(h1);

  if (is_positive_) {
    *(uint32_t*)out = h1;
  } else {
    *(int32_t*)out = h1;
  }
}

Status MurmurHash3::Compute(OpKernelContext* ctx) const {
  const Tensor* keys = ctx->Input<Tensor>(0);
  ORT_ENFORCE(keys);

  const TensorShape& input_shape = keys->Shape();
  Tensor* output_tensor = ctx->Output(0, input_shape);

  const MLDataType keys_type = keys->DataType();
  const bool is_string = utils::IsDataTypeString(keys_type);

  const auto input_element_bytes = keys->DataType()->Size();
  const auto output_element_bytes = output_tensor->DataType()->Size();
  const int64_t input_count = input_shape.Size();
  // Output type is inferred by the inference function and it can be of two types int32_t and uint32_t
  // however, all is needed is a ptr that can step 4 bytes at a time and for that reason we choose
  // raw data casted to a type of choice.
  ORT_ENFORCE(sizeof(uint32_t) == output_element_bytes, "Invalid assumption of output element size");
  auto output = reinterpret_cast<uint32_t*>(output_tensor->MutableDataRaw());

  if (is_string) {
    auto input = keys->Data<std::string>();
    const auto input_end = input + input_count;
    while (input != input_end) {
      MurmurHash3_x86_32(input->c_str(),
                         static_cast<int>(input->length()),
                         seed_,
                         output);
      ++input;
      ++output;
    }
  } else {
    auto input = reinterpret_cast<const uint32_t*>(keys->DataRaw());
    const auto input_end = input + input_count;
    while (input != input_end) {
      MurmurHash3_x86_32(input,
                         static_cast<int>(input_element_bytes),
                         seed_,
                         output);
      ++input;
      ++output;
    }
  }
  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
