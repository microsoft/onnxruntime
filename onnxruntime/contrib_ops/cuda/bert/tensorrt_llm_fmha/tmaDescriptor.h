/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#if USE_TENSORRT_LLM_FMHA

//namespace onnxruntime {
namespace tensorrt_llm
{
namespace kernels
{

// TMA desc type.
typedef enum
{
    TILED = 0,
    IM2COL
} cudaTmaDescType;

// TMA swizzle type.
typedef enum
{
    SWIZZLE_DISABLED,
    SWIZZLE_32B,
    SWIZZLE_64B,
    SWIZZLE_128B,
    SWIZZLE_MAX
} cudaTmaDescSwizzle;

typedef enum
{
    BARRIER64,
    BARRIER128
} cudaTmaDescBarrier;

// TMA interleave type.
typedef enum
{
    INTERLEAVE_DISABLED,
    INTERLEAVE_16B,
    INTERLEAVE_32B,
    INTERLEAVE_MAX
} cudaTmaDescInterleave;

// TMA L2 sector promotion.
typedef enum
{
    PROMOTION_DISABLED = 0,
    PROMOTION_64B,
    PROMOTION_128B,
    PROMOTION_256B
} cudaTmaDescPromotion;

// TMA data type.
typedef enum
{
    U8 = 0,
    U16,
    U32,
    S32,
    U64,
    S64,
    F16_RN,
    F32_RN,
    F32_FTZ_RN,
    F64_RN,
    BF16_RN,
    FORMAT_MAX
} cudaTmaDescFormat;

// TMA cache control.
typedef enum
{
    PREFETCH,      // Prefetch tma descriptor using global memory address
    INVALIDATE,    // Invalidate tma descriptor in l2 cache
    INVALIDATE_ALL // Invalidate tma descriptor and all elements in l2 cache line
} cudaTmaDescCacheCtrl;

// TMA OOB fill modes.
typedef enum
{
    TENSOR_ZFILL,
    TENSOR_CFILL
} cudaTmaDescOobFillMode;

constexpr uint64_t k_max_tensor_size = (1llu << 36);
constexpr uint64_t k_max_tensor_stride = (1llu << 36);
constexpr uint64_t k_max_block_size = 256llu;
constexpr uint64_t k_max_traversal_stride = (1llu << 3);

constexpr uint64_t k_min_tensor_size = 1llu;
constexpr uint64_t k_min_tensor_stride = 0llu;
constexpr uint64_t k_min_block_size = 1llu;
constexpr uint64_t k_min_traversal_stride = 1llu;

constexpr uint32_t k_max_cta_id = (1 << 6) - 1;

// The 512 bit of descriptor for tiled mode.
typedef struct
{
    uint64_t tensor_common0;
    uint32_t tensor_common1;

    uint32_t tensor_stride_lower[4]; //< 36b of 64b with 4B aligned
    uint32_t tensor_stride_upper;
    uint32_t tensor_size[5];         //< value -1
    uint32_t traversal_stride_box_0; //< packed 3b (-1)

    uint32_t box_size_end;
} cudaTmaDescTiled;

// The 512 bit of descritptro for im2col mode.
typedef struct
{
    uint64_t tensor_common0;
    uint32_t tensor_common1;

    uint32_t tensor_stride_lower[4];
    uint32_t tensor_stride_upper;
    uint32_t tensor_size[5];
    uint32_t traversal_stride_range_c;

    uint32_t box_corner_dhw;
    uint32_t range_ndhw;
} cudaTmaDescIm2Col;

// TMA desc size
constexpr uint32_t TMA_DESC_SIZE_IN_BYTE = 64;

// TMA desc
typedef struct alignas(64)
{
    uint64_t data[8];
} cudaTmaDesc;

////////////

// manage TMA descriptor host code.
// allocate, deallocate and manipulate tma desc in the host
// copy the tma descriptor from host code to device code
// Multiple TMA desc, one desc per batch.
// Device desc ptr should be allocated outside the class and reused
template <
    // number of dimensions.
    int NUM_DIMS>
class Multiple_tma_descriptor
{
public:
    // ctor
    Multiple_tma_descriptor(int batch_size_)
        : batch_size(batch_size_)
    {
        if (batch_size > 0)
        {
            // allocate host memory
            desc_ptr_h = new cudaTmaDesc[batch_size];
            // make sure all bit fields are zeros.
            memset(desc_ptr_h, 0, sizeof(cudaTmaDesc) * batch_size);
        }
    }

    // ctor
    Multiple_tma_descriptor() = default;

    // destructor.
    ~Multiple_tma_descriptor()
    {
        if (batch_size > 0)
        {
            // deallocate host memory
            delete[] desc_ptr_h;
        }
    }

    // set the desctriptor.
    int set_tma_desctriptor(
        // ptr to gmem
        const void* gmem_ptr,
        // format is really data_type in TMA terminology.
        cudaTmaDescFormat format,
        // interleave mode.
        cudaTmaDescInterleave interleave,
        // swizzle mode.
        cudaTmaDescSwizzle swizzle,
        // L2 sector promotion.
        cudaTmaDescPromotion promotion, const uint32_t (&tensor_size_array)[NUM_DIMS],
        const uint64_t (&tensor_stride_array)[NUM_DIMS - 1], const uint32_t (&traversal_stride_array)[NUM_DIMS],
        const uint32_t (&box_size_array)[NUM_DIMS],
        // OOB fill mode.
        uint32_t fill_oob,
        // FP32 to TF32 conversion.
        uint32_t round_to_tf32,
        // index to desc.
        int batch_idx)
    {

        set_tensor_common_0(&desc_ptr_h[batch_idx], reinterpret_cast<uint64_t>(gmem_ptr));
        set_tensor_common_1(
            &desc_ptr_h[batch_idx], TILED, NUM_DIMS, format, interleave, swizzle, fill_oob, round_to_tf32, promotion);

        set_tensor_stride(&desc_ptr_h[batch_idx], tensor_stride_array);
        set_tensor_size(&desc_ptr_h[batch_idx], tensor_size_array);

        set_traversal_stride_tiled(&desc_ptr_h[batch_idx], traversal_stride_array);

        set_box_size(&desc_ptr_h[batch_idx], box_size_array);
        return 0;
    }

    // set the desctriptor.
    int set_tma_desctriptor(
        // ptr to gmem
        const void* gmem_ptr,
        // format is really data_type in TMA terminology.
        cudaTmaDescFormat format,
        // interleave mode.
        cudaTmaDescInterleave interleave,
        // swizzle mode.
        cudaTmaDescSwizzle swizzle,
        // L2 sector promotion.
        cudaTmaDescPromotion promotion, const uint32_t (&tensor_size_array)[NUM_DIMS],
        const uint64_t (&tensor_stride_array)[NUM_DIMS - 1], const uint32_t (&traversal_stride_array)[NUM_DIMS],
        const uint32_t (&box_size_array)[NUM_DIMS],
        // OOB fill mode.
        uint32_t fill_oob,
        // FP32 to TF32 conversion.
        uint32_t round_to_tf32,
        // index to desc.
        cudaTmaDesc* desc_ptr = nullptr)
    {

        set_tensor_common_0(desc_ptr, reinterpret_cast<uint64_t>(gmem_ptr));
        set_tensor_common_1(desc_ptr, TILED, NUM_DIMS, format, interleave, swizzle, fill_oob, round_to_tf32, promotion);

        set_tensor_stride(desc_ptr, tensor_stride_array);
        set_tensor_size(desc_ptr, tensor_size_array);

        set_traversal_stride_tiled(desc_ptr, traversal_stride_array);

        set_box_size(desc_ptr, box_size_array);
        return 0;
    }

    // copy the desc to device memory
    void copy_to_device(void* desc_ptr_d_, cudaStream_t stream = 0)
    {
        cudaMemcpyAsync(desc_ptr_d_, desc_ptr_h, TMA_DESC_SIZE_IN_BYTE * batch_size, cudaMemcpyHostToDevice, stream);
    }

    // get desc in host
    cudaTmaDesc get_desc_in_host(int batch_idx) const
    {
        return desc_ptr_h[batch_idx];
    }

private:
    void set_tensor_common_0(cudaTmaDesc* p_desc, uint64_t addr)
    {
        cudaTmaDescTiled* desc = reinterpret_cast<cudaTmaDescTiled*>(p_desc);
        desc->tensor_common0 = 0;
        desc->tensor_common0 |= (addr);
    }

    void set_tensor_common_1(cudaTmaDesc* p_desc, cudaTmaDescType desc_type, uint32_t dims, cudaTmaDescFormat format,
        cudaTmaDescInterleave interleave, cudaTmaDescSwizzle swizzle, uint32_t fill, uint32_t f32_to_tf32,
        cudaTmaDescPromotion promotion)
    {
        cudaTmaDescTiled* desc = reinterpret_cast<cudaTmaDescTiled*>(p_desc);

        desc->tensor_common1 = 0;
        desc->tensor_common1 |= desc_type == TILED ? 0x0 : 0x1;

        constexpr uint32_t VERSION_SHIFT = 1;
        constexpr uint32_t VERSION_BITS = 3;
        desc->tensor_common1 |= (1u << VERSION_SHIFT);

        constexpr uint32_t DIM_BITS = 3;
        constexpr uint32_t DIM_SHIFT = VERSION_SHIFT + VERSION_BITS;
        constexpr uint32_t DIM_MASK = (1u << DIM_BITS) - 1;
        desc->tensor_common1 |= ((dims - 1) & DIM_MASK) << DIM_SHIFT;

        constexpr uint32_t FORMAT_BITS = 4;
        constexpr uint32_t FORMAT_SHIFT = DIM_SHIFT + DIM_BITS;
        constexpr uint32_t FORMAT_MASK = (1u << FORMAT_BITS) - 1;
        desc->tensor_common1 |= (static_cast<uint32_t>(format) & FORMAT_MASK) << FORMAT_SHIFT;

        constexpr uint32_t INTERLEAVE_BITS = 2;
        constexpr uint32_t INTERLEAVE_SHIFT = FORMAT_SHIFT + FORMAT_BITS;
        constexpr uint32_t INTERLEAVE_MASK = (1u << INTERLEAVE_BITS) - 1;
        desc->tensor_common1 |= (static_cast<uint32_t>(interleave) & INTERLEAVE_MASK) << INTERLEAVE_SHIFT;

        constexpr uint32_t SWIZZLE_BITS = 2;
        constexpr uint32_t SWIZZLE_SHIFT = INTERLEAVE_SHIFT + INTERLEAVE_BITS;
        constexpr uint32_t SWIZZLE_MASK = (1u << SWIZZLE_BITS) - 1;
        desc->tensor_common1 |= (static_cast<uint32_t>(swizzle) & SWIZZLE_MASK) << SWIZZLE_SHIFT;

        constexpr uint32_t FILL_BITS = 1;
        constexpr uint32_t FILL_SHIFT = SWIZZLE_SHIFT + SWIZZLE_BITS;
        constexpr uint32_t FILL_MASK = (1u << FILL_BITS) - 1;
        desc->tensor_common1 |= (static_cast<uint32_t>(fill) & FILL_MASK) << FILL_SHIFT;

        constexpr uint32_t F32_TO_TF32_BITS = 1;
        constexpr uint32_t F32_TO_TF32_SHIFT = FILL_SHIFT + FILL_BITS;
        constexpr uint32_t F32_TO_TF32_MASK = (1u << F32_TO_TF32_BITS) - 1;
        desc->tensor_common1 |= (static_cast<uint32_t>(f32_to_tf32) & F32_TO_TF32_MASK) << F32_TO_TF32_SHIFT;

        constexpr uint32_t PROMOTION_BITS = 2;
        constexpr uint32_t PROMOTION_SHIFT = F32_TO_TF32_SHIFT + F32_TO_TF32_BITS;
        constexpr uint32_t PROMOTION_MASK = (1u << PROMOTION_BITS) - 1;
        desc->tensor_common1 |= (static_cast<uint32_t>(promotion) & PROMOTION_MASK) << PROMOTION_SHIFT;
    }

    // note that tensor stride has 1 less dim.
    void set_tensor_stride(cudaTmaDesc* p_desc, const uint64_t (&tensor_stride_array)[NUM_DIMS - 1])
    {
        cudaTmaDescTiled* desc = reinterpret_cast<cudaTmaDescTiled*>(p_desc);

        constexpr uint32_t TENSOR_STRIDE_UPPER_BITS = 4;
        constexpr uint32_t TENSOR_STRIDE_UPPER_MASK = (1u << TENSOR_STRIDE_UPPER_BITS) - 1;

        for (uint32_t i = 0; i < NUM_DIMS - 1; i++)
        {
            desc->tensor_stride_lower[i] = 0u;
            uint64_t tensor_stride_lower_64b = (tensor_stride_array[i] >> 4) & 0xFFFFFFFFlu;
            desc->tensor_stride_lower[i] = static_cast<uint32_t>(tensor_stride_lower_64b);
        }
        desc->tensor_stride_upper = 0u;

        for (uint32_t i = 0; i < NUM_DIMS - 1; i++)
        {
            uint64_t tensor_stride_temp = tensor_stride_array[i];
            tensor_stride_temp = tensor_stride_temp >> 4;
            uint64_t tensor_stride_upper = tensor_stride_temp >> 32;
            uint32_t tensor_stride_upper_32b = static_cast<uint32_t>(tensor_stride_upper);
            desc->tensor_stride_upper
                |= ((tensor_stride_upper_32b & TENSOR_STRIDE_UPPER_MASK) << (i * TENSOR_STRIDE_UPPER_BITS));
        }
    }

    void set_tensor_size(cudaTmaDesc* p_desc, const uint32_t (&tensor_size_array)[NUM_DIMS])
    {
        cudaTmaDescTiled* desc = reinterpret_cast<cudaTmaDescTiled*>(p_desc);
        for (uint32_t dim = 0; dim < NUM_DIMS; dim++)
        {
            desc->tensor_size[dim] = tensor_size_array[dim] - 1;
        }
    }

    void set_traversal_stride_tiled(cudaTmaDesc* p_desc, const uint32_t (&traversal_stride_array)[NUM_DIMS])
    {
        cudaTmaDescTiled* desc = reinterpret_cast<cudaTmaDescTiled*>(p_desc);

        desc->traversal_stride_box_0 = 0;

        constexpr uint32_t TRAVERSAL_STRIDE_BITS = 3;
        constexpr uint32_t TRAVERSAL_STRIDE_MASK = (1u << TRAVERSAL_STRIDE_BITS) - 1;

        for (uint32_t dim = 0; dim < NUM_DIMS; dim++)
        {
            uint32_t traversal_stride = traversal_stride_array[dim] - 1;
            traversal_stride = (traversal_stride & TRAVERSAL_STRIDE_MASK) << (dim * TRAVERSAL_STRIDE_BITS);
            desc->traversal_stride_box_0 |= traversal_stride;
        }
    }

    void set_box_size(cudaTmaDesc* p_desc, const uint32_t (&box_size_array)[NUM_DIMS])
    {
        cudaTmaDescTiled* desc = reinterpret_cast<cudaTmaDescTiled*>(p_desc);

        desc->box_size_end = 0;

        constexpr uint32_t BOX_SIZE_BITS = 8;
        constexpr uint32_t BOX_SIZE_MASK = (1 << BOX_SIZE_BITS) - 1;

        if (NUM_DIMS > 1)
        {
            uint32_t box_size_0 = box_size_array[0] - 1;
            box_size_0 = box_size_0 & BOX_SIZE_MASK;
            box_size_0 = box_size_0 << 24;
            desc->traversal_stride_box_0 |= box_size_0;
        }

        for (uint32_t dim = 1; dim < NUM_DIMS; dim++)
        {
            uint32_t box_size = box_size_array[dim] - 1;
            box_size = box_size & BOX_SIZE_MASK;
            box_size = box_size << ((dim - 1) * BOX_SIZE_BITS);
            desc->box_size_end |= box_size;
        }
    }

    void set_traversal_stride_im2col(cudaTmaDesc* p_desc, uint32_t* p_traversal_stride, uint32_t dims)
    {

        cudaTmaDescIm2Col* desc = reinterpret_cast<cudaTmaDescIm2Col*>(p_desc);

        desc->traversal_stride_range_c = 0;

        constexpr uint32_t TRAVERSAL_STRIDE_BITS = 3;
        constexpr uint32_t TRAVERSAL_STRIDE_MASK = (1u << (TRAVERSAL_STRIDE_BITS + 1)) - 1;

        for (uint32_t dim = 0; dim < dims; dim++)
        {
            uint32_t traversal_stride = p_traversal_stride[dim] - 1;
            traversal_stride = (traversal_stride & TRAVERSAL_STRIDE_MASK) << (dim * TRAVERSAL_STRIDE_BITS);
            desc->traversal_stride_range_c |= traversal_stride;
        }
    }

    void set_range_c(cudaTmaDesc* p_desc, uint32_t range_c)
    {
        cudaTmaDescIm2Col* desc = reinterpret_cast<cudaTmaDescIm2Col*>(p_desc);

        constexpr uint32_t RANGE_C_BITS = 8;
        constexpr uint32_t RANGE_C_MASK = (1u << RANGE_C_BITS) - 1;

        range_c = range_c & RANGE_C_MASK;
        desc->traversal_stride_range_c |= ((range_c - 1) << 24);
    }

    void set_box_corner_dhw(cudaTmaDesc* p_desc, uint32_t* p_base_corner, uint32_t* p_far_corner, uint32_t dims)
    {
        cudaTmaDescIm2Col* desc = reinterpret_cast<cudaTmaDescIm2Col*>(p_desc);

        desc->box_corner_dhw = 0;

        uint32_t box_base_corner = 0, box_far_corner = 0;
        uint32_t box_corner_dhw = 0;

        if (dims == 3)
        {
            constexpr uint32_t BOX_CORNER_BITS = 16;
            constexpr uint32_t BOX_CORNER_MASK = (1u << BOX_CORNER_BITS) - 1;

            box_base_corner = p_base_corner[0] & BOX_CORNER_MASK;
            box_far_corner = p_far_corner[0] & BOX_CORNER_MASK;
        }

        if (dims == 4)
        {
            constexpr uint32_t BOX_CORNER_BITS = 8;
            constexpr uint32_t BOX_CORNER_MASK = (1u << BOX_CORNER_BITS) - 1;

            box_base_corner = p_base_corner[0] & BOX_CORNER_MASK;
            box_base_corner |= ((p_base_corner[1] & BOX_CORNER_MASK) << BOX_CORNER_BITS);

            box_far_corner = p_far_corner[0] & BOX_CORNER_MASK;
            box_far_corner |= ((p_far_corner[1] & BOX_CORNER_MASK) << BOX_CORNER_BITS);
        }

        if (dims == 5)
        {
            constexpr uint32_t BOX_CORNER_BITS = 5;
            constexpr uint32_t BOX_CORNER_MASK = (1u << BOX_CORNER_BITS) - 1;

            box_base_corner = p_base_corner[0] & BOX_CORNER_MASK;
            box_base_corner |= ((p_base_corner[1] & BOX_CORNER_MASK) << BOX_CORNER_BITS);
            box_base_corner |= ((p_base_corner[2] & BOX_CORNER_MASK) << (2 * BOX_CORNER_BITS));

            box_far_corner = p_far_corner[0] & BOX_CORNER_MASK;
            box_far_corner |= ((p_far_corner[1] & BOX_CORNER_MASK) << BOX_CORNER_BITS);
            box_far_corner |= ((p_far_corner[2] & BOX_CORNER_MASK) << (2 * BOX_CORNER_BITS));
        }

        box_corner_dhw = box_base_corner;
        box_corner_dhw |= (box_far_corner << 16);

        desc->box_corner_dhw = box_corner_dhw;
    }

    void set_range_ndhw(cudaTmaDesc* p_desc, uint32_t ndhw)
    {
        cudaTmaDescIm2Col* desc = reinterpret_cast<cudaTmaDescIm2Col*>(p_desc);

        desc->range_ndhw = 0;

        constexpr uint32_t RANGE_NDHW_BITS = 10;
        constexpr uint32_t RANGE_NDHW_MASK = (1u << RANGE_NDHW_BITS) - 1;

        desc->range_ndhw = ((ndhw - 1) & RANGE_NDHW_MASK);
    }

    // The TMA descriptor. Each is of 512 bit.
    cudaTmaDesc* desc_ptr_h;
    // The TMA descriptor on the device memory.
    cudaTmaDesc* desc_ptr_d;
    // Number of batches
    int batch_size = 0;
};

} // namespace kernels
} // namespace tensorrt_llm
//}

#endif
