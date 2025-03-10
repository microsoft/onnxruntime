/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:
    pooling_fp16.cpp

Abstract:
    This module implements the pooling operation for fp16
    tensors in NHWC format.
--*/

#include "mlasi.h"

#include "fp16_common.h"

#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED


template <typename AggregationType>
typename AggregationType::CtxType
PoolCreateContext(size_t KernelSize);

template <typename AggregationType>
MLAS_FLOAT16X8
PoolInit16x8();

template <typename AggregationType>
MLAS_FLOAT16X4
PoolInit16x4();

template <typename AggregationType>
MLAS_FLOAT16X8
PoolAggregate16x8(MLAS_FLOAT16X8 agg, MLAS_FLOAT16X8 element);

template <typename AggregationType>
MLAS_FLOAT16X4
PoolAggregate16x4(MLAS_FLOAT16X4 agg, MLAS_FLOAT16X4 element);

template <typename AggregationType>
MLAS_FLOAT16X8
PoolSummary16x8(MLAS_FLOAT16X8 agg, typename AggregationType::CtxType context);

template <typename AggregationType>
MLAS_FLOAT16X4
PoolSummary16x4(MLAS_FLOAT16X4 agg, typename AggregationType::CtxType context);


struct MaxPoolAggregation {
    typedef size_t CtxType; // useless type to satisfy compilers
};

template <>
MLAS_FORCEINLINE
size_t
PoolCreateContext<MaxPoolAggregation>(size_t KernelSize)
{
    MLAS_UNREFERENCED_PARAMETER(KernelSize);
    return 0;
}

template<>
MLAS_FORCEINLINE
MLAS_FLOAT16X8
PoolInit16x8<MaxPoolAggregation>()
{
    // lowest fp16 -65504.0f
    return MlasBroadcastFloat16x8(0xfbff);
}

template <>
MLAS_FORCEINLINE
MLAS_FLOAT16X4
PoolInit16x4<MaxPoolAggregation>()
{
    // lowest fp16 -65504.0f
    return MlasBroadcastFloat16x4(0xfbff);
}

template<>
MLAS_FORCEINLINE 
MLAS_FLOAT16X8
PoolAggregate16x8<MaxPoolAggregation>(MLAS_FLOAT16X8 agg, MLAS_FLOAT16X8 element)
{
    return MlasMaximumFloat16x8(agg, element);
}

template<>
MLAS_FORCEINLINE 
MLAS_FLOAT16X4
PoolAggregate16x4<MaxPoolAggregation>(MLAS_FLOAT16X4 agg, MLAS_FLOAT16X4 element)
{
    return MlasMaximumFloat16x4(agg, element);
}

template<>
MLAS_FORCEINLINE 
MLAS_FLOAT16X8
PoolSummary16x8<MaxPoolAggregation>(MLAS_FLOAT16X8 agg, size_t size)
{
    MLAS_UNREFERENCED_PARAMETER(size);
    return agg;
}

template<>
MLAS_FORCEINLINE 
MLAS_FLOAT16X4
PoolSummary16x4<MaxPoolAggregation>(MLAS_FLOAT16X4 agg, size_t size)
{
    MLAS_UNREFERENCED_PARAMETER(size);
    return agg;
}

struct AveragePoolAggregation {
    typedef MLAS_FLOAT16X8 CtxType;
};

template<>
MLAS_FLOAT16X8
PoolCreateContext<AveragePoolAggregation>(size_t KernelSize)
{
    return MlasBroadcastFloat16x8(MLAS_Float2Half(float(KernelSize)));
}


template <>
MLAS_FORCEINLINE MLAS_FLOAT16X8
PoolInit16x8<AveragePoolAggregation>()
{
    return MlasZeroFloat16x8();
}

template <>
MLAS_FORCEINLINE MLAS_FLOAT16X4
PoolInit16x4<AveragePoolAggregation>()
{
    return MlasZeroFloat16x4();
}


template <>
MLAS_FORCEINLINE MLAS_FLOAT16X8
PoolAggregate16x8<AveragePoolAggregation>(MLAS_FLOAT16X8 agg, MLAS_FLOAT16X8 element)
{
    return MlasAddFloat16x8(agg, element);
}

template <>
MLAS_FORCEINLINE MLAS_FLOAT16X4
PoolAggregate16x4<AveragePoolAggregation>(MLAS_FLOAT16X4 agg, MLAS_FLOAT16X4 element)
{
    return MlasAddFloat16x4(agg, element);
}

template <>
MLAS_FORCEINLINE MLAS_FLOAT16X8
PoolSummary16x8<AveragePoolAggregation>(MLAS_FLOAT16X8 agg, MLAS_FLOAT16X8 context)
{
    return MlasDivFloat16x8(agg, context);
}

template <>
MLAS_FORCEINLINE MLAS_FLOAT16X4
PoolSummary16x4<AveragePoolAggregation>(MLAS_FLOAT16X4 agg, MLAS_FLOAT16X8 context)
{
    return MlasDivFloat16x4(agg, MlasToLowHalfFloat16x4(context));
}


template<typename AggregationType>
MLAS_FORCEINLINE
void
MlasPoolFp16HWC(
    const _mlas_fp16_* const* Input,
    _mlas_fp16_* Output,
    size_t Channels,
    size_t OutputCount,
    size_t KernelSize
    )
{
    while (OutputCount > 0) {
        size_t ChannelOffset = 0;
        size_t c = Channels;

        while (c >= 32) {
            MLAS_FLOAT16X8 MaximumVector0 = PoolInit16x8<AggregationType>();
            MLAS_FLOAT16X8 MaximumVector1 = MaximumVector0;
            MLAS_FLOAT16X8 MaximumVector2 = MaximumVector0;
            MLAS_FLOAT16X8 MaximumVector3 = MaximumVector0;
            size_t cnt = 0;

            for (size_t k = 0; k < KernelSize; k++) {
                if (Input[k] == nullptr) {
                    continue;
                }
                MLAS_FLOAT16X8 InputVector0 = MlasLoadFloat16x8(&Input[k][ChannelOffset]);
                MLAS_FLOAT16X8 InputVector1 = MlasLoadFloat16x8(&Input[k][ChannelOffset + 8]);
                MLAS_FLOAT16X8 InputVector2 = MlasLoadFloat16x8(&Input[k][ChannelOffset + 16]);
                MLAS_FLOAT16X8 InputVector3 = MlasLoadFloat16x8(&Input[k][ChannelOffset + 24]);

                MaximumVector0 = PoolAggregate16x8<AggregationType>(MaximumVector0, InputVector0);
                MaximumVector1 = PoolAggregate16x8<AggregationType>(MaximumVector1, InputVector1);
                MaximumVector2 = PoolAggregate16x8<AggregationType>(MaximumVector2, InputVector2);
                MaximumVector3 = PoolAggregate16x8<AggregationType>(MaximumVector3, InputVector3);
                cnt++;
            }
            typename AggregationType::CtxType context = PoolCreateContext<AggregationType>(cnt);
            MaximumVector0 = PoolSummary16x8<AggregationType>(MaximumVector0, context);
            MaximumVector1 = PoolSummary16x8<AggregationType>(MaximumVector1, context);
            MaximumVector2 = PoolSummary16x8<AggregationType>(MaximumVector2, context);
            MaximumVector3 = PoolSummary16x8<AggregationType>(MaximumVector3, context);

            MlasStoreFloat16x8(&Output[0], MaximumVector0);
            MlasStoreFloat16x8(&Output[8], MaximumVector1);
            MlasStoreFloat16x8(&Output[16], MaximumVector2);
            MlasStoreFloat16x8(&Output[24], MaximumVector3);

            Output += 32;
            ChannelOffset += 32;
            c -= 32;
        }

        if (c >= 16) {
            MLAS_FLOAT16X8 MaximumVector0 = PoolInit16x8<AggregationType>();
            MLAS_FLOAT16X8 MaximumVector1 = MaximumVector0;
            size_t cnt = 0;

            for (size_t k = 0; k < KernelSize; k++) {
                if (Input[k] == nullptr) {
                    continue;
                }
                MLAS_FLOAT16X8 InputVector0 = MlasLoadFloat16x8(&Input[k][ChannelOffset]);
                MLAS_FLOAT16X8 InputVector1 = MlasLoadFloat16x8(&Input[k][ChannelOffset + 8]);

                MaximumVector0 = PoolAggregate16x8<AggregationType>(MaximumVector0, InputVector0);
                MaximumVector1 = PoolAggregate16x8<AggregationType>(MaximumVector1, InputVector1);
                cnt++;
            }
            typename AggregationType::CtxType context = PoolCreateContext<AggregationType>(cnt);
            MaximumVector0 = PoolSummary16x8<AggregationType>(MaximumVector0, context);
            MaximumVector1 = PoolSummary16x8<AggregationType>(MaximumVector1, context);

            MlasStoreFloat16x8(&Output[0], MaximumVector0);
            MlasStoreFloat16x8(&Output[8], MaximumVector1);

            Output += 16;
            ChannelOffset += 16;
            c -= 16;
        }

        if (c >= 8) {
            MLAS_FLOAT16X8 MaximumVector0 = PoolInit16x8<AggregationType>();
            size_t cnt = 0;

            for (size_t k = 0; k < KernelSize; k++) {
                if (Input[k] == nullptr) {
                    continue;
                }
                MLAS_FLOAT16X8 InputVector0 = MlasLoadFloat16x8(&Input[k][ChannelOffset]);
                MaximumVector0 = PoolAggregate16x8<AggregationType>(MaximumVector0, InputVector0);
                cnt++;
            }
            typename AggregationType::CtxType context = PoolCreateContext<AggregationType>(cnt);
            MaximumVector0 = PoolSummary16x8<AggregationType>(MaximumVector0, context);

            MlasStoreFloat16x8(&Output[0], MaximumVector0);

            Output += 8;
            ChannelOffset += 8;
            c -= 8;
        }

        if (c >= 4) {
            MLAS_FLOAT16X4 MaximumVector0 = PoolInit16x4<AggregationType>();
            size_t cnt = 0;

            for (size_t k = 0; k < KernelSize; k++) {
                if (Input[k] == nullptr) {
                    continue;
                }
                MLAS_FLOAT16X4 InputVector0 = MlasLoadFloat16x4(&Input[k][ChannelOffset]);
                MaximumVector0 = PoolAggregate16x4<AggregationType>(MaximumVector0, InputVector0);
                cnt++;
            }
            typename AggregationType::CtxType context = PoolCreateContext<AggregationType>(cnt);
            MaximumVector0 = PoolSummary16x4<AggregationType>(MaximumVector0, context);

            MlasStoreFloat16x4(&Output[0], MaximumVector0);

            Output += 4;
            ChannelOffset += 4;
            c -= 4;
        }

        if (c > 0) {
            // possible over read by 7 bytes
            MLAS_FLOAT16X4 MaximumVector0 = PoolInit16x4<AggregationType>();
            size_t cnt = 0;

            for (size_t k = 0; k < KernelSize; k++) {
                if (Input[k] == nullptr) {
                    continue;
                }
                MLAS_FLOAT16X4 InputVector0 = MlasLoadFloat16x4(&Input[k][ChannelOffset]);
                MaximumVector0 = PoolAggregate16x4<AggregationType>(MaximumVector0, InputVector0);
                cnt++;
            }
            typename AggregationType::CtxType context = PoolCreateContext<AggregationType>(cnt);
            MaximumVector0 = PoolSummary16x4<AggregationType>(MaximumVector0, context);

            MlasStorePartialFloat16x4(&Output[0], MaximumVector0, c);
            Output += c;
        }

        Input += KernelSize;
        OutputCount -= 1;
    }
}


void
MLASCALL
MlasNhwcMaxPool(
    const MLAS_FP16* const* Input,
    MLAS_FP16* Output,
    size_t Channels,
    size_t OutputCount,
    size_t KernelSize
    )
{
    const _mlas_fp16_* const* input_ptr = reinterpret_cast<const _mlas_fp16_* const*>(Input);
    _mlas_fp16_* output_ptr = reinterpret_cast<_mlas_fp16_*>(Output);
    MlasPoolFp16HWC<MaxPoolAggregation>(input_ptr, output_ptr, Channels, OutputCount, KernelSize);
}


void
MLASCALL
MlasNhwcAvgPool(
    const MLAS_FP16* const* Input,
    MLAS_FP16* Output,
    size_t Channels,
    size_t OutputCount,
    size_t KernelSize
    )
{
    const _mlas_fp16_* const* input_ptr = reinterpret_cast<const _mlas_fp16_* const*>(Input);
    _mlas_fp16_* output_ptr = reinterpret_cast<_mlas_fp16_*>(Output);
    MlasPoolFp16HWC<AveragePoolAggregation>(input_ptr, output_ptr, Channels, OutputCount, KernelSize);
}


#endif  // MLAS_F16VEC_INTRINSICS_SUPPORTED
