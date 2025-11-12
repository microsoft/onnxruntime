// Forked from
// https://devicesasg.visualstudio.com/PerceptiveShell/_git/PerceptiveShell?version=GC8dbf8ccf0881d607494c6147e208af7415d1a169&path=/src/perceptiveshell_private/perceptiveshell_private/crypto.c
/*
    THIS MODULE IS DEFINED WITH OPTIMISATIONS ON.
    We forego the extra heap, etc checks, and in return get much faster loading of models in debug.
    This is probably a good trade-off.
    This is a "C" file, not because C is faster than C++ (it is not!)
    It is a "C" file because CMake forces us to do it, as it only has one global set of flags per language.
*/

#if defined(_M_X64) && defined(_WIN64) // use 256 bit acceleration if possible, but only works on Windows/X64
#include <assert.h>
#include <immintrin.h>
#include <isa_availability.h>
#include <stdlib.h>

typedef __m256 BulkType;
#define __m256_num_bytes (32) // __m256 is 32 bytes
#define __m256_mask (__m256_num_bytes - 1)
#define __m256_ISA_name __IA_SUPPORT_VECTOR256
// Return if the pointer is aligned to the correct byte-boundary
static inline int is_aligned(void* p)
{
    return !((__int64)p & __m256_mask);
}

// ConstructAlignedKey : Fill a block of BulkType[key_size] with __m256_num_bytes repeated copies of the key.
// The alignment of the keys must coincide with the alignment of the data to be en-/decrypted.
// tgt is the aligned target buffer containing the key copies
// src is the single unaligned key
// offset is the mis-alignment of the data
// len is the key length
// The idea is:
// KKK|KKKKKKK|KKKKKKK|KKKKKKK|KKKKKKK|...|KKKK
// ...........|Data buffer ...
// <--offset->|
// The key KKKKKKK needs to be aligned with the data and filled in multiple times left and right of the
// offset point.
static void ConstructAlignedKey(BulkType* tgt, const char* src, long long offset, long long len)
{
    const long long lower = -offset / len;                // start index = - (number of copies left of the offset point)
    const long long upper = __m256_num_bytes + lower - 1; // stop index for last full copy
    char* tgt_bytes = (char*)tgt;

    const long long top_fragment =
        offset + lower * len; // length of the key fragment wrapped around from the end to the start

    // copy the wrapped top fragment
    memcpy(tgt_bytes, src + len - top_fragment, top_fragment);

    for (long long idx = lower; idx < upper; ++idx)
    {
        // make all full copies
        memcpy(tgt_bytes + offset + idx * len, src, len);
    }

    // copy the last partial key fragment at the end
    memcpy(tgt_bytes + offset + upper * len, src, len - top_fragment);
}

// XOR fill:
// key must be byte sizeof(BulkType)-aligned
// tgt must be byte sizeof(BulkType)-aligned
// NOTE:
static void XORFill(const BulkType* key, int key_offset, char* tgt, long long key_sz, long long tgt_sz)
{
    assert(is_aligned(tgt));
    BulkType* tgt_blocks = (BulkType*)tgt;
    for (long long idx = 0; idx < tgt_sz; ++idx)
    {
        tgt_blocks[idx] = _mm256_xor_ps(tgt_blocks[idx], key[key_offset++]);
        key_offset = key_offset == key_sz ? 0 : key_offset;
    }
}

/*
* Specialised decrypt_impl for architectures with BlockType support
* Use the default method, until we hit an aligned block of memory
* Then create a repeating block of keys.
* XOR this block repeatedly against the tgt, using SIMD
* Finally complete the last few non-aligned bytes (if any), again using the standard method.

Advantages: We gain SIMD XOR, two load instruction per cache-line, one store instruction per cache-line. This is the
most efficient way to use memory. Disadvantages: Setting up the repeating block will cost time, may not be a speed-gain
if the tgt-size is small.
*/
void decrypt_impl(const char* key, long long key_size, char* data, long long size)
{
    if (!__check_isa_support(__m256_ISA_name, 0))
    {
        for (long long i = 0; i < size; i++)
        {
            data[i] ^= key[i % key_size];
        }
    }
    else
    {
        long long i = 0;
        for (; i < size; i++)
        {
            if (is_aligned(&data[i]))
            {
                break;
            }
            data[i] ^= key[i % key_size];
        }

        const long long num_SIMD = (size - i) / __m256_num_bytes;
        if (num_SIMD)
        {
            __m256* key_buffer = (__m256*)_aligned_malloc( // _aligned_malloc is Windows API
                __m256_num_bytes * key_size, // allocate space for our repeated keys, align to the type boundary
                __m256_num_bytes);

            const long long offset = (__m256_num_bytes - i) % __m256_num_bytes;
            ConstructAlignedKey(key_buffer, key, offset, key_size);
            XORFill(key_buffer, i == 0 ? 0 : 1, &data[i], key_size, num_SIMD);
            i += num_SIMD * __m256_num_bytes;
            _aligned_free(key_buffer);
        }

        for (; i < size; i++)
        {
            data[i] ^= key[i % key_size];
        }
    }
}

#else
void decrypt_impl(const char* key, long long key_size, char* data, long long size)
{
    for (long long i = 0; i < size; i++)
    {
        data[i] ^= key[i % key_size];
    }
}
#endif // x64&WIN32 speedup.
