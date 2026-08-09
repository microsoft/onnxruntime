/*++

Copyright (c) Microsoft Corporation. All rights reserved.
Copyright (c) 2019, 2022, Oracle and/or its affiliates. All rights reserved.

Licensed under the MIT License.

Module Name:

    reorder.cpp

Abstract:

    This module implements routines to reorder buffers to and from blocked
    formats.

--*/

#include "mlasi.h"

//
// Define the parameters to execute segments of a NCHW output reordering
// operation on worker threads.
//

struct MLAS_REORDER_OUTPUT_NCHW_BLOCK {
    ptrdiff_t TargetThreadCount;
    const float* S;
    float* D;
    size_t OutputChannels;
    size_t OutputSize;
    size_t TasksCount;
};

MLAS_FORCEINLINE
void
MlasReorderGatherFloat32x4(
    const float* S,
    float* D,
    size_t GatherStride
    )
/*++

Routine Description:

    This routine gathers floats from the source buffer and writes a vector to
    the destination buffer.

Arguments:

    S - Supplies the address of the source buffer.

    D - Supplies the address of the destination buffer.

    GatherStride - Supplies the stride to read elements from the source buffer.

Return Value:

    None.

--*/
{
#if defined(MLAS_SSE41_INTRINSICS)
    __m128 v = _mm_load_ss(&S[0 * GatherStride]);
    v = _mm_insert_ps(v, _mm_load_ss(&S[1 * GatherStride]), 0x10);
    v = _mm_insert_ps(v, _mm_load_ss(&S[2 * GatherStride]), 0x20);
    v = _mm_insert_ps(v, _mm_load_ss(&S[3 * GatherStride]), 0x30);

    _mm_storeu_ps(D, v);
#else
    float f0 = S[0 * GatherStride];
    float f1 = S[1 * GatherStride];
    float f2 = S[2 * GatherStride];
    float f3 = S[3 * GatherStride];

    D[0] = f0;
    D[1] = f1;
    D[2] = f2;
    D[3] = f3;
#endif
}

MLAS_FORCEINLINE
void
MlasReorderScatterFloat32x4(
    const float* S,
    float* D,
    size_t ScatterStride
    )
/*++

Routine Description:

    This routine scatters a vector read from the source buffer to the
    destination buffer.

Arguments:

    S - Supplies the address of the source buffer.

    D - Supplies the address of the destination buffer.

    ScatterStride - Supplies the stride to write elements to the destination
        buffer.

Return Value:

    None.

--*/
{
#if defined(MLAS_SSE41_INTRINSICS) || defined(MLAS_NEON_INTRINSICS)
    MLAS_FLOAT32X4 v = MlasLoadFloat32x4(S);

    MlasStoreLaneFloat32x4<0>(&D[ScatterStride * 0], v);
    MlasStoreLaneFloat32x4<1>(&D[ScatterStride * 1], v);
    MlasStoreLaneFloat32x4<2>(&D[ScatterStride * 2], v);
    MlasStoreLaneFloat32x4<3>(&D[ScatterStride * 3], v);
#else
    float f0 = S[0];
    float f1 = S[1];
    float f2 = S[2];
    float f3 = S[3];

    D[ScatterStride * 0] = f0;
    D[ScatterStride * 1] = f1;
    D[ScatterStride * 2] = f2;
    D[ScatterStride * 3] = f3;
#endif
}

MLAS_FORCEINLINE
void
MlasReorderTransposeFloat32x4x4(
    const float* S,
    float* D,
    size_t GatherStride,
    size_t ScatterStride
    )
/*++

Routine Description:

    This routine transposes a 4x4 float matrix read from the source buffer and
    writes the result to the destination buffer.

Arguments:

    S - Supplies the address of the source buffer.

    D - Supplies the address of the destination buffer.

    GatherStride - Supplies the stride to read elements from the source buffer.

    ScatterStride - Supplies the stride to write vectors to the destination
        buffer.

Return Value:

    None.

--*/
{
#if defined(MLAS_SSE2_INTRINSICS)
    MLAS_FLOAT32X4 v[4];
    MLAS_FLOAT32X4 t[4];

    v[0] = MlasLoadFloat32x4(&S[GatherStride * 0]);
    v[1] = MlasLoadFloat32x4(&S[GatherStride * 1]);
    v[2] = MlasLoadFloat32x4(&S[GatherStride * 2]);
    v[3] = MlasLoadFloat32x4(&S[GatherStride * 3]);

    t[0] = _mm_unpacklo_ps(v[0], v[1]);
    t[2] = _mm_unpackhi_ps(v[0], v[1]);
    t[1] = _mm_unpacklo_ps(v[2], v[3]);
    t[3] = _mm_unpackhi_ps(v[2], v[3]);

    v[0] = _mm_movelh_ps(t[0], t[1]);
    v[1] = _mm_movehl_ps(t[1], t[0]);
    v[2] = _mm_movelh_ps(t[2], t[3]);
    v[3] = _mm_movehl_ps(t[3], t[2]);

    MlasStoreFloat32x4(&D[ScatterStride * 0], v[0]);
    MlasStoreFloat32x4(&D[ScatterStride * 1], v[1]);
    MlasStoreFloat32x4(&D[ScatterStride * 2], v[2]);
    MlasStoreFloat32x4(&D[ScatterStride * 3], v[3]);
#elif  defined(MLAS_LSX_INTRINSICS)

    MLAS_FLOAT32X4 v[4];
    MLAS_FLOAT32X4 t[4];

    v[0] = MlasLoadFloat32x4(&S[GatherStride * 0]);
    v[1] = MlasLoadFloat32x4(&S[GatherStride * 1]);
    v[2] = MlasLoadFloat32x4(&S[GatherStride * 2]);
    v[3] = MlasLoadFloat32x4(&S[GatherStride * 3]);

    t[0] = (__m128)__lsx_vilvl_w((__m128i)v[1], (__m128i)v[0]);
    t[2] = (__m128)__lsx_vilvh_w((__m128i)v[1], (__m128i)v[0]);
    t[1] = (__m128)__lsx_vilvl_w((__m128i)v[3], (__m128i)v[2]);
    t[3] = (__m128)__lsx_vilvh_w((__m128i)v[3], (__m128i)v[2]);


    v[0] = (__m128)__lsx_vpickev_d((__m128i) t[1],(__m128i) t[0]);
    v[1] = (__m128)__lsx_vpickod_d((__m128i) t[1],(__m128i) t[0]);
    v[2] = (__m128)__lsx_vpickev_d((__m128i) t[3],(__m128i) t[2]);
    v[3] = (__m128)__lsx_vpickod_d((__m128i) t[3],(__m128i) t[2]);

    MlasStoreFloat32x4(&D[ScatterStride * 0], v[0]);
    MlasStoreFloat32x4(&D[ScatterStride * 1], v[1]);
    MlasStoreFloat32x4(&D[ScatterStride * 2], v[2]);
    MlasStoreFloat32x4(&D[ScatterStride * 3], v[3]);
#else
    MlasReorderScatterFloat32x4(&S[GatherStride * 0], &D[0], ScatterStride);
    MlasReorderScatterFloat32x4(&S[GatherStride * 1], &D[1], ScatterStride);
    MlasReorderScatterFloat32x4(&S[GatherStride * 2], &D[2], ScatterStride);
    MlasReorderScatterFloat32x4(&S[GatherStride * 3], &D[3], ScatterStride);
#endif
}

void
MLASCALL
MlasReorderInputNchw(
    const float* S,
    float* D,
    size_t InputChannels,
    size_t InputSize
    )
/*++

Routine Description:

    This routine reorders an input buffer from NCHW to NCHWc format.

Arguments:

    S - Supplies the address of the source tensor.

    D - Supplies the address of the destination tensor.

    InputChannels - Supplies the number of NCHW channels.

    InputSize - Supplies the spatial input size of the tensors.

Return Value:

    None.

--*/
{
    const size_t BlockSize = MlasNchwcGetBlockSize();

    const MLAS_FLOAT32X4 ZeroFloat32x4 = MlasZeroFloat32x4();

    //
    // Iterate over BlockSize batches of the input channels.
    //

    for (size_t i = InputChannels; i > 0;) {

        const size_t InputChannelsThisIteration = std::min(i, BlockSize);
        i -= InputChannelsThisIteration;

        const float* s = S;
        float* d = D;
        size_t InputSizeRemaining = InputSize;

        for (; InputSizeRemaining >= 4; InputSizeRemaining -= 4) {

            const float* ss = s;
            float* dd = d;
            size_t bc = 0;

            for (; bc < InputChannelsThisIteration; bc += 4) {
                MlasReorderTransposeFloat32x4x4(ss, dd, InputSize, BlockSize);
                ss += 4 * InputSize;
                dd += 4;
            }

            for (; bc < BlockSize; bc += 4) {
                MlasStoreFloat32x4(&dd[BlockSize * 0], ZeroFloat32x4);
                MlasStoreFloat32x4(&dd[BlockSize * 1], ZeroFloat32x4);
                MlasStoreFloat32x4(&dd[BlockSize * 2], ZeroFloat32x4);
                MlasStoreFloat32x4(&dd[BlockSize * 3], ZeroFloat32x4);
                dd += 4;
            }

            s += 4;
            d += 4 * BlockSize;
        }

        for (; InputSizeRemaining > 0; InputSizeRemaining--) {

            const float* ss = s;
            float* dd = d;
            size_t bc = 0;

            for (; bc < InputChannelsThisIteration; bc += 4) {
                MlasReorderGatherFloat32x4(ss, dd, InputSize);
                ss += 4 * InputSize;
                dd += 4;
            }

            for (; bc < BlockSize; bc += 4) {
                MlasStoreFloat32x4(dd, ZeroFloat32x4);
                dd += 4;
            }

            s += 1;
            d += BlockSize;
        }

        S += BlockSize * InputSize;
        D += BlockSize * InputSize;
    }
}

void
MLASCALL
MlasReorderInputNhwc(
    const float* S,
    float* D,
    size_t InputChannels,
    size_t RowCount,
    size_t FullRowCount
    )
/*++

Routine Description:

    This routine reorders an input buffer from NHWC to NCHWc format.

Arguments:

    S - Supplies the address of the source tensor.

    D - Supplies the address of the destination tensor.

    InputChannels - Supplies the number of NHWC channels.

    RowCount - Supplies the number of NHWC rows to process. This number may be
        less than FullRowCount to support threaded operation.

    FullRowCount - Supplies the total number of NHWC rows per image.

Return Value:

    None.

--*/
{
    const size_t BlockSize = MlasNchwcGetBlockSize();

    //
    // Iterate over batches of the input size to improve locality.
    //

    for (size_t OuterRowCountRemaining = RowCount; OuterRowCountRemaining > 0; ) {

        constexpr size_t OuterRowCountBatch = 32;

        const size_t OuterRowCountThisIteration = std::min(OuterRowCountRemaining, OuterRowCountBatch);
        OuterRowCountRemaining -= OuterRowCountThisIteration;

        //
        // Iterate over BlockSize batches of the input channels.
        //

        const float* s = S;
        float* d = D;

        for (size_t i = InputChannels; i > 0;) {

            const size_t InputChannelsThisIteration = std::min(i, BlockSize);
            i -= InputChannelsThisIteration;

            const float* ss = s;
            float* dd = d;
            size_t InnerRowCountRemaining = OuterRowCountThisIteration;

            if (InputChannelsThisIteration == BlockSize) {

                if (BlockSize == 8) {

                    while (InnerRowCountRemaining-- > 0) {

                        MLAS_FLOAT32X4 v0 = MlasLoadFloat32x4(&ss[0]);
                        MLAS_FLOAT32X4 v1 = MlasLoadFloat32x4(&ss[4]);

                        MlasStoreFloat32x4(&dd[0], v0);
                        MlasStoreFloat32x4(&dd[4], v1);

                        ss += InputChannels;
                        dd += 8;
                    }

                } else {

                    while (InnerRowCountRemaining-- > 0) {

                        MLAS_FLOAT32X4 v0 = MlasLoadFloat32x4(&ss[0]);
                        MLAS_FLOAT32X4 v1 = MlasLoadFloat32x4(&ss[4]);
                        MLAS_FLOAT32X4 v2 = MlasLoadFloat32x4(&ss[8]);
                        MLAS_FLOAT32X4 v3 = MlasLoadFloat32x4(&ss[12]);

                        MlasStoreFloat32x4(&dd[0], v0);
                        MlasStoreFloat32x4(&dd[4], v1);
                        MlasStoreFloat32x4(&dd[8], v2);
                        MlasStoreFloat32x4(&dd[12], v3);

                        ss += InputChannels;
                        dd += 16;
                    }
                }

            } else {

                size_t BlockPadding = BlockSize - InputChannelsThisIteration;

                while (InnerRowCountRemaining-- > 0) {

                    std::copy_n(ss, InputChannelsThisIteration, dd);
                    std::fill_n(dd + InputChannelsThisIteration, BlockPadding, 0.0f);

                    ss += InputChannels;
                    dd += BlockSize;
                }
            }

            s += InputChannelsThisIteration;
            d += BlockSize * FullRowCount;
        }

        S += InputChannels * OuterRowCountThisIteration;
        D += BlockSize * OuterRowCountThisIteration;
    }
}

void
MlasReorderOutputNchwThreaded(
    void* Context,
    ptrdiff_t Index
    )
/*++

Routine Description:

    This routine is invoked from a worker thread to execute a segment of a
    NCHW output reordering operation.

Arguments:

    Context - Supplies the pointer to the context for the threaded operation.

    Index - Supplies the current index of the threaded operation.

Return Value:

    None.

--*/
{
    const auto* WorkBlock = (MLAS_REORDER_OUTPUT_NCHW_BLOCK*)Context;

    const size_t OutputChannels = WorkBlock->OutputChannels;
    const size_t OutputSize = WorkBlock->OutputSize;
    const float* S = WorkBlock->S;
    float* D =  WorkBlock->D;

    const size_t BlockSize = MlasNchwcGetBlockSize();
    const size_t TasksPerBatch = size_t(ceil(((float)OutputChannels) / BlockSize));
    const size_t LastTaskInBatchIndex = TasksPerBatch - 1;

    //
    // Compute the range of task indices to use for this thread.
    //

    size_t TaskStart;
    size_t TasksRemaining;

    MlasPartitionWork(Index, WorkBlock->TargetThreadCount, WorkBlock->TasksCount,
        &TaskStart, &TasksRemaining);

    size_t TaskEnd = TaskStart + TasksRemaining;
    //
    // Rebase the pointers to the source and destination buffers for this thread.
    //

    size_t FirstBatchIndex = TaskStart / TasksPerBatch;
    size_t FirstTaskInBatchIndex = TaskStart % TasksPerBatch;
    S += BlockSize * OutputSize * (FirstBatchIndex * TasksPerBatch + FirstTaskInBatchIndex);
    D += OutputSize * (FirstBatchIndex * OutputChannels + BlockSize * FirstTaskInBatchIndex);

    //
    // Transpose NCHWc blocks associated with tasks in the range [TaskStart, TaskEnd)
    // from the source buffer to the destination buffer.
    //

    for (size_t t = TaskStart; t < TaskEnd; t++) {
        size_t TaskInBatchIndex = t % TasksPerBatch;

        const size_t OutputChannelsThisIteration = (TaskInBatchIndex < LastTaskInBatchIndex) ?
            BlockSize : OutputChannels - BlockSize * LastTaskInBatchIndex;
        const size_t AlignedOutputChannelsThisIteration = OutputChannelsThisIteration & (~3);

        const float* s = S;
        float* d = D;
        size_t OutputSizeRemaining = OutputSize;

        for (; OutputSizeRemaining >= 4; OutputSizeRemaining -= 4) {

            const float* ss = s;
            float* dd = d;
            size_t bc = 0;

            for (; bc < AlignedOutputChannelsThisIteration; bc += 4) {
                MlasReorderTransposeFloat32x4x4(ss, dd, BlockSize, OutputSize);
                ss += 4;
                dd += 4 * OutputSize;
            }

            for (; bc < OutputChannelsThisIteration; bc += 1) {
                MlasReorderGatherFloat32x4(ss, dd, BlockSize);
                ss += 1;
                dd += OutputSize;
            }

            s += 4 * BlockSize;
            d += 4;
        }

        for (; OutputSizeRemaining > 0; OutputSizeRemaining--) {

            const float* ss = s;
            float* dd = d;
            size_t bc = 0;

            for (; bc < AlignedOutputChannelsThisIteration; bc += 4) {
                MlasReorderScatterFloat32x4(ss, dd, OutputSize);
                ss += 4;
                dd += 4 * OutputSize;
            }

            for (; bc < OutputChannelsThisIteration; bc += 1) {
                *dd = *ss++;
                dd += OutputSize;
            }

            s += BlockSize;
            d += 1;
        }

        S += BlockSize * OutputSize;
        D += OutputChannelsThisIteration * OutputSize;
    }
}


void
MLASCALL
MlasReorderOutputNchw(
    const int64_t* OutputShape,
    const float* S,
    float* D,
    MLAS_THREADPOOL* ThreadPool
    )
/*++

Routine Description:

    This routine reorders an output buffer from NCHWc to NCHW format.

Arguments:

    OutputShape - Supplies the shape of the output tensor.

    S - Supplies the address of the source tensor.

    D - Supplies the address of the destination tensor.

Return Value:

    None.

--*/
{
    MLAS_REORDER_OUTPUT_NCHW_BLOCK WorkBlock;

    //
    // Capture the NCHW reorder output operation parameters to the work block.
    //

    WorkBlock.S = S;
    WorkBlock.D = D;
    WorkBlock.OutputChannels = size_t(OutputShape[1]);
    WorkBlock.OutputSize = size_t(OutputShape[2]) * size_t(OutputShape[3]);

    const size_t BlockSize = MlasNchwcGetBlockSize();
    const size_t TasksPerBatch = size_t(ceil(((float)WorkBlock.OutputChannels) / BlockSize));
    const size_t BatchCount = size_t(OutputShape[0]);
    const size_t TasksCount = BatchCount * TasksPerBatch;
    WorkBlock.TasksCount = TasksCount;

    //
    // Schedule the operation across a set of worker threads if the output
    // tensor is sufficienly large. Limit the number of threads to at least
    // the number of available tasks.
    //

    ptrdiff_t TargetThreadCount = 1;
    const size_t BufferSize = BatchCount * WorkBlock.OutputChannels * WorkBlock.OutputSize;
    if (BufferSize > 1024 && TasksCount > 1) {
        TargetThreadCount = MlasGetMaximumThreadCount(ThreadPool);
        if (size_t(TargetThreadCount) > TasksCount) {
            TargetThreadCount = ptrdiff_t(TasksCount);
        }
    }
    WorkBlock.TargetThreadCount = TargetThreadCount;

    MlasExecuteThreaded(MlasReorderOutputNchwThreaded, &WorkBlock, TargetThreadCount, ThreadPool);
}

void
MLASCALL
MlasReorderOutputNhwc(
    const int64_t* OutputShape,
    const float* S,
    float* D
    )
/*++

Routine Description:

    This routine reorders an output buffer from NCHWc to NHWC format.

Arguments:

    OutputShape - Supplies the shape of the output tensor.

    S - Supplies the address of the source tensor.

    D - Supplies the address of the destination tensor.

Return Value:

    None.

--*/
{
    const size_t BlockSize = MlasNchwcGetBlockSize();

    const size_t BatchCount = size_t(OutputShape[0]);
    const size_t OutputChannels = size_t(OutputShape[3]);
    const size_t OutputSize = size_t(OutputShape[1]) * size_t(OutputShape[2]);

    const size_t AlignedOutputChannels = (OutputChannels + BlockSize - 1) & ~(BlockSize - 1);

    //
    // Copy NCHWc blocks from the source buffer to the destination buffer.
    //

    for (size_t batch = 0; batch < BatchCount; batch++) {

        const float* s = S;
        size_t OutputSizeRemaining = OutputSize;

        for (; OutputSizeRemaining > 0; OutputSizeRemaining--) {

            const float* ss = s;

            for (size_t o = OutputChannels; o > 0;) {

                const size_t OutputChannelsThisIteration = std::min(o, BlockSize);
                const size_t AlignedOutputChannelsThisIteration = OutputChannelsThisIteration & (~3);
                o -= OutputChannelsThisIteration;

                size_t bc = 0;

                for (; bc < AlignedOutputChannelsThisIteration; bc += 4) {
                    MlasStoreFloat32x4(&D[bc], MlasLoadFloat32x4(&ss[bc]));
                }

                for (; bc < OutputChannelsThisIteration; bc += 1) {
                    D[bc] = ss[bc];
                }

                ss += BlockSize * OutputSize;
                D += OutputChannelsThisIteration;
            }

            s += BlockSize;
        }

        S += AlignedOutputChannels * OutputSize;
    }
}

void
MLASCALL
MlasReorderFilterOIHWBiBo(
    const int64_t* FilterShape,
    const float* S,
    float* D
    )
/*++

Routine Description:

    This routine reorders a filter buffer from OIHW to OIHWBiBo format.

Arguments:

    FilterShape - Supplies the shape of the filter tensor.

    S - Supplies the address of the source tensor.

    D - Supplies the address of the destination tensor.

Return Value:

    None.

--*/
{
    const size_t BlockSize = MlasNchwcGetBlockSize();

    const size_t OutputChannels = size_t(FilterShape[0]);
    const size_t InputChannels = size_t(FilterShape[1]);
    const size_t KernelHeight = size_t(FilterShape[2]);
    const size_t KernelWidth = size_t(FilterShape[3]);

    const size_t KernelSize = KernelHeight * KernelWidth;
    const size_t InputStride = InputChannels * KernelSize;

    const MLAS_FLOAT32X4 ZeroFloat32x4 = MlasZeroFloat32x4();

    //
    // Transform the filter tensor from format OIHW to OIHWBiBo:
    //
    //  OutputChannelBlock[0] = {
    //      InputChannelBlock[0] = {
    //          Kernel[0][0] = {
    //              InputChannel[0] = { filter[0 filter[1] ... filter[BlockSize-1] },
    //              InputChannel[1] = { filter[0 filter[1] ... filter[BlockSize-1] },
    //              ...
    //              InputChannel[BlockSize-1] = { filter[0] filter[1] ... filter[BlockSize-1] },
    //          },
    //          Kernel[0][1] = {
    //              ...
    //          },
    //          ...
    //          Kernel[KernelHeight-1][KernelWidth-1] = {
    //              ...
    //          },
    //      },
    //      InputChannelBlock[BlockSize] = {
    //          ...
    //      },
    //      ...
    //      InputChannelBlock[InputChannels-BlockSize] = {
    //          ...
    //      },
    //  },
    //  OutputChannelBlock[BlockSize] = {
    //      ...
    //  },
    //  OutputChannelBlock[OutputChannels-BlockSize] = {
    //      ...
    //  };
    //

    //
    // Iterate over BlockSize batches of the output channels.
    //
    // The final batch may be less than BlockSize, but must be a multiple of 4.
    // The unaligned count results in zero padding below.
    //

    for (size_t o = OutputChannels; o > 0;) {

        const size_t OutputChannelsThisIteration = std::min(o, BlockSize);
        const size_t AlignedOutputChannelsThisIteration = OutputChannelsThisIteration & (~3);
        o -= OutputChannelsThisIteration;

        //
        // Iterate over BlockSize batches of the input channels.
        //
        // The final batch may be less than BlockSize, but must be a multiple
        // of 4.
        //

        const float* S_InputChannels = S;

        for (size_t i = InputChannels; i > 0;) {

            const size_t InputChannelsThisIteration = std::min(i, BlockSize);
            i -= InputChannelsThisIteration;

            //
            // Iterate over each index of the kernel.
            //

            const float* S_KernelSize = S_InputChannels;

            for (size_t k = 0; k < KernelSize; k++) {

                //
                // Construct a filter block of BlockSize by BlockSize.
                //

                const float* S_BlockSize = S_KernelSize;

                for (size_t bi = 0; bi < InputChannelsThisIteration; bi++) {

                    //
                    // Transpose from the source filter buffer to the destination
                    // buffer. Zero pad the filter block if the output channels
                    // is not block aligned.
                    //

                    const float* s = S_BlockSize;
                    size_t bo = 0;

                    for (; bo < AlignedOutputChannelsThisIteration; bo += 4) {
                        MlasReorderGatherFloat32x4(s, D, InputStride);
                        s += 4 * InputStride;
                        D += 4;
                    }

                    for (; bo < OutputChannelsThisIteration; bo += 1) {
                        *D++ = *s;
                        s += InputStride;
                    }

                    for (; bo < BlockSize; bo += 1) {
                        *D++ = 0.0f;
                    }

                    S_BlockSize += KernelSize;
                }

                for (size_t z = 0; z < (BlockSize - InputChannelsThisIteration) * (BlockSize / 4); z++) {
                    MlasStoreFloat32x4(D, ZeroFloat32x4);
                    D += 4;
                }

                S_KernelSize += 1;
            }

            S_InputChannels += BlockSize * KernelSize;
        }

        S += BlockSize * InputStride;
    }
}

void
MLASCALL
MlasReorderFilterOIHWBo(
    const int64_t* FilterShape,
    const float* S,
    float* D
    )
/*++

Routine Description:

    This routine reorders a filter buffer from OIHW to OIHWBo format.

Arguments:

    FilterShape - Supplies the shape of the filter tensor.

    S - Supplies the address of the source tensor.

    D - Supplies the address of the destination tensor.

Return Value:

    None.

--*/
{
    const size_t BlockSize = MlasNchwcGetBlockSize();

    const size_t OutputChannels = size_t(FilterShape[0]);
    const size_t InputChannels = size_t(FilterShape[1]);
    const size_t KernelHeight = size_t(FilterShape[2]);
    const size_t KernelWidth = size_t(FilterShape[3]);

    const size_t KernelSize = KernelHeight * KernelWidth;
    const size_t InputStride = InputChannels * KernelSize;

    //
    // Transform the filter tensor from format OIHW to OIHWBo:
    //
    //  OutputChannelBlock[0] = {
    //      InputChannel[0] = {
    //          Kernel[0][0] = filter[0 filter[1] ... filter[BlockSize-1] },
    //          Kernel[0][1] = { filter[0 filter[1] ... filter[BlockSize-1] },
    //          ...
    //          Kernel[KernelHeight-1][KernelWidth-1] = { filter[0 filter[1] ... filter[BlockSize-1] },
    //      },
    //      InputChannel[1] = {
    //          ...
    //      },
    //      ...
    //      InputChannel[InputChannels-1] = {
    //          ...
    //      },
    //  },
    //  OutputChannelBlock[BlockSize] = {
    //      ...
    //  },
    //  OutputChannelBlock[OutputChannels-BlockSize] = {
    //      ...
    //  };
    //

    //
    // Iterate over BlockSize batches of the output channels.
    //
    // The final batch may be less than BlockSize, but must be a multiple of 4.
    // The unaligned count results in zero padding below.
    //

    for (size_t o = OutputChannels; o > 0;) {

        const size_t OutputChannelsThisIteration = std::min(o, BlockSize);
        const size_t AlignedOutputChannelsThisIteration = OutputChannelsThisIteration & (~3);
        o -= OutputChannelsThisIteration;

        //
        // Iterate over each of the input channels.
        //

        const float* S_InputChannels = S;

        for (size_t i = 0; i < InputChannels; i += 1) {

            //
            // Iterate over each index of the kernel.
            //

            const float* S_KernelSize = S_InputChannels;

            for (size_t k = 0; k < KernelSize; k++) {

                //
                // Transpose a float[4] from the source filter buffer to the
                // destination buffer. Zero pad the filter block if the output
                // channels is not block aligned.
                //

                const float* s = S_KernelSize;
                size_t bo = 0;

                for (; bo < AlignedOutputChannelsThisIteration; bo += 4) {
                    MlasReorderGatherFloat32x4(s, D, InputStride);
                    s += 4 * InputStride;
                    D += 4;
                }

                for (; bo < OutputChannelsThisIteration; bo += 1) {
                    *D++ = *s;
                    s += InputStride;
                }

                for (; bo < BlockSize; bo += 1) {
                    *D++ = 0.0f;
                }

                S_KernelSize += 1;
            }

            S_InputChannels += KernelSize;
        }

        S += BlockSize * InputStride;
    }
}
