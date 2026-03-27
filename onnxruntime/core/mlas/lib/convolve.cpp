/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    convolve.cpp

Abstract:

    This module implements the convolution operation.

--*/

#include "mlasi.h"

#include <mutex>
#include <unordered_map>

//
// Define the number of working buffer elements required per thread.
//

#define MLAS_CONV_WORKING_BUFFER_SIZE_PER_THREAD \
    (MLAS_SGEMM_STRIDEN * MLAS_SGEMM_STRIDEK)

//
// Define the parameters to execute segments of a convolution operation on
// worker threads.
//

struct MLAS_CONV_WORK_BLOCK {
    const MLAS_CONV_PARAMETERS* Parameters;
    const float* Input;
    const float* Filter;
    const float* Bias;
    float* WorkingBuffer;
    float* Output;
    struct SEGMENT {
        size_t StartN;
        size_t CountN;
    } Segments[MLAS_MAXIMUM_THREAD_COUNT];
    ptrdiff_t TargetThreadCount;
};

static
void
MlasConvDepthwiseMultiplier2FloatCHW(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    const float* Filter,
    float* Output
    )
/*++

Routine Description:

    This routine implements a narrow direct convolution kernel for grouped
    convolutions where the number of input channels per group is one and the
    number of output channels per group is two.

    The first implementation intentionally targets the MobileCLIP projection
    case and operates directly from the standard OIHW filter layout.

Arguments:

    Parameters - Supplies the convolution parameters.

    Input - Supplies the input tensor for one batch/group slice.

    Filter - Supplies the filter tensor for one group in OIHW layout.

    Output - Supplies the output tensor for one batch/group slice.

Return Value:

    None.

--*/
{
    const size_t InputHeight = Parameters->InputShape[0];
    const size_t InputWidth = Parameters->InputShape[1];
    const size_t OutputHeight = Parameters->OutputShape[0];
    const size_t OutputWidth = Parameters->OutputShape[1];
    const size_t KernelHeight = Parameters->KernelShape[0];
    const size_t KernelWidth = Parameters->KernelShape[1];
    const size_t DilationHeight = Parameters->DilationShape[0];
    const size_t DilationWidth = Parameters->DilationShape[1];
    const size_t StrideHeight = Parameters->StrideShape[0];
    const size_t StrideWidth = Parameters->StrideShape[1];
    const ptrdiff_t PaddingTop = static_cast<ptrdiff_t>(Parameters->Padding[0]);
    const ptrdiff_t PaddingLeft = static_cast<ptrdiff_t>(Parameters->Padding[1]);
    const size_t OutputSize = Parameters->OutputSize;

    const float* Filter0 = Filter;
    const float* Filter1 = Filter + KernelHeight * KernelWidth;
    float* Output0 = Output;
    float* Output1 = Output + OutputSize;

#if defined(MLAS_TARGET_AMD64)
    if (KernelHeight == 7 && KernelWidth == 7 &&
        DilationHeight == 1 && DilationWidth == 1 &&
        StrideHeight == 2 && StrideWidth == 2 &&
        PaddingTop == 3 && PaddingLeft == 3 &&
        GetMlasPlatform().ConvNchwFloatKernel == MlasConvNchwFloatKernelAvx512F) {
        MlasConvDepthwiseWithMultiplierFloatCHWKernel7x7Stride2DepthMultiplier2Avx512F(
            Input,
            InputHeight,
            InputWidth,
            Filter,
            Output,
            OutputHeight,
            OutputWidth,
            Parameters->Beta);
        return;
    }
#endif

    for (size_t oh = 0; oh < OutputHeight; ++oh) {
        const ptrdiff_t InputOriginY = static_cast<ptrdiff_t>(oh * StrideHeight) - PaddingTop;

        for (size_t ow = 0; ow < OutputWidth; ++ow) {
            const ptrdiff_t InputOriginX = static_cast<ptrdiff_t>(ow * StrideWidth) - PaddingLeft;
            const size_t OutputIndex = oh * OutputWidth + ow;

            float Acc0 = (Parameters->Beta == 0.0f) ? 0.0f : (Output0[OutputIndex] * Parameters->Beta);
            float Acc1 = (Parameters->Beta == 0.0f) ? 0.0f : (Output1[OutputIndex] * Parameters->Beta);

            for (size_t kh = 0; kh < KernelHeight; ++kh) {
                const ptrdiff_t ih = InputOriginY + static_cast<ptrdiff_t>(kh * DilationHeight);
                if (ih < 0 || ih >= static_cast<ptrdiff_t>(InputHeight)) {
                    continue;
                }

                const size_t InputRowOffset = static_cast<size_t>(ih) * InputWidth;
                const size_t FilterRowOffset = kh * KernelWidth;

                for (size_t kw = 0; kw < KernelWidth; ++kw) {
                    const ptrdiff_t iw = InputOriginX + static_cast<ptrdiff_t>(kw * DilationWidth);
                    if (iw < 0 || iw >= static_cast<ptrdiff_t>(InputWidth)) {
                        continue;
                    }

                    const float InputValue = Input[InputRowOffset + static_cast<size_t>(iw)];
                    const size_t FilterIndex = FilterRowOffset + kw;

                    Acc0 += InputValue * Filter0[FilterIndex];
                    Acc1 += InputValue * Filter1[FilterIndex];
                }
            }

            Output0[OutputIndex] = Acc0;
            Output1[OutputIndex] = Acc1;
        }
    }
}

void
MlasConvIm2Col(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    float* ColumnBuffer,
    size_t k,
    size_t CountK,
    size_t n,
    size_t CountN
    )
/*++

Routine Description:

    This routine converts the input image to a set of convolution patches
    appropriate for use with a GEMM operation.

    This implementation supports sampling a portion of the convolution
    patches. This avoids the need to allocate very large buffers to store
    all of the convolution patches at once, when the underlying GEMM
    implementation will already break up the operation into panels. Multiple
    threads can also be used to process different portions of the image.

Arguments:

    Parameters - Supplies the structure that contains the convolution
        parameters.

    Input - Supplies the input tensor.

    ColumnBuffer - Supplies the buffer to receive the convolution patches.

    k - Supplies the K to begin sampling the convolution patches.

    CountK - Supplies the count of K to sample for the convolution patches.

    n - Supplies the N to begin sampling the convolution patches.

    CountN - Supplies the count of N to sample for the convolution patches.

Return Value:

    None.

--*/
{
    constexpr size_t HeightShapeIndex = 0;
    constexpr size_t WidthShapeIndex = 1;

    const size_t OutputWidth = Parameters->OutputShape[WidthShapeIndex];

    const size_t StrideHeight = Parameters->StrideShape[HeightShapeIndex];
    const size_t StrideWidth = Parameters->StrideShape[WidthShapeIndex];

    const size_t nx = (n % OutputWidth);
    const size_t ny = (n / OutputWidth);

    const size_t OriginInputX = nx * StrideWidth;
    const size_t OriginInputY = ny * StrideHeight;

    size_t OutputCountX = OutputWidth - nx;

    const size_t InputHeight = Parameters->InputShape[HeightShapeIndex];
    const size_t InputWidth = Parameters->InputShape[WidthShapeIndex];
    const size_t InputSize = Parameters->InputSize;

    const size_t KernelHeight = Parameters->KernelShape[HeightShapeIndex];
    const size_t KernelWidth = Parameters->KernelShape[WidthShapeIndex];

    size_t kx = (k % KernelWidth);
    size_t ky = (k / KernelWidth) % KernelHeight;

    Input = Input + (k / (KernelHeight * KernelWidth)) * InputSize;

    const size_t DilationHeight = Parameters->DilationShape[HeightShapeIndex];
    const size_t DilationWidth = Parameters->DilationShape[WidthShapeIndex];

    const size_t PaddingLeftY = Parameters->Padding[HeightShapeIndex];
    const size_t PaddingLeftX = Parameters->Padding[WidthShapeIndex];

    for (size_t EndingK = k + CountK; k < EndingK; k++) {

        size_t CountX = OutputCountX;
        size_t InputY = (ky * DilationHeight) + OriginInputY - PaddingLeftY;
        const size_t RowInitialInputX = (kx * DilationWidth) - PaddingLeftX;
        size_t InitialInputX = RowInitialInputX + OriginInputX;
        size_t RemainingN = CountN;

        do {

            if (CountX > RemainingN) {
                CountX = RemainingN;
            }

            RemainingN -= CountX;

            //
            // Check if the input is in the top/bottom padding region.
            //

            if (InputY < InputHeight) {

                size_t InputX = InitialInputX;
                const float* InputRow = &Input[InputY * InputWidth];

                do {

                    //
                    // Check if the input is in the left/right padding region.
                    //

                    if (InputX >= InputWidth) {

                        *ColumnBuffer++ = 0;
                        InputX += StrideWidth;
                        CountX--;

                    } else if (StrideWidth == 1) {

                        //
                        // Copy input elements to the column buffer.
                        //

                        size_t CountCopyX = InputWidth - InputX;

                        if (CountCopyX > CountX) {
                            CountCopyX = CountX;
                        }

                        CountX -= CountCopyX;

                        while (CountCopyX >= 4) {
                            MlasStoreFloat32x4(ColumnBuffer, MlasLoadFloat32x4(&InputRow[InputX]));
                            ColumnBuffer += 4;
                            InputX += 4;
                            CountCopyX -= 4;
                        }

                        while (CountCopyX > 0) {
                            *ColumnBuffer++ = InputRow[InputX++];
                            CountCopyX--;
                        }

                    } else if (InputX + CountX * StrideWidth <= InputWidth) {

                        do {
                            *ColumnBuffer++ = InputRow[InputX];
                            InputX += StrideWidth;
                        } while (--CountX > 0);

                    } else {

                        do {
                            *ColumnBuffer++ = (InputX < InputWidth) ? InputRow[InputX] : 0;
                            InputX += StrideWidth;
                        } while (--CountX > 0);
                    }

                } while (CountX > 0);

            } else {

                //
                // The entire input row is in the padding region.
                //

                MLAS_FLOAT32X4 ZeroFloat32x4 = MlasZeroFloat32x4();

                while (CountX >= 4) {
                    MlasStoreFloat32x4(ColumnBuffer, ZeroFloat32x4);
                    ColumnBuffer += 4;
                    CountX -= 4;
                }

                while (CountX > 0) {
                    MlasStoreLaneFloat32x4<0>(ColumnBuffer, ZeroFloat32x4);
                    ColumnBuffer++;
                    CountX--;
                }
            }

            CountX = OutputWidth;
            InputY += StrideHeight;
            InitialInputX = RowInitialInputX;

        } while (RemainingN > 0);

        //
        // Advance the kernel indices and advance to the next channel if the
        // entire kernel is complete.
        //

        if (++kx == KernelWidth) {

            if (++ky == KernelHeight) {

                Input += InputSize;

                ky = 0;
            }

            kx = 0;
        }
    }
}

static bool
MlasConvTileTuningEnabledByEnv()
/*++

Routine Description:

    This routine checks whether the experimental convolution tile tuning path
    is enabled via environment variable.

Return Value:

    Returns true if the tile tuning path is enabled.

--*/
{
    static const bool Enabled = []() {
        constexpr const char* EnvVarName = "ORT_MLAS_CONV_7X7_TILE_TUNING_ENABLE";

#if defined(_WIN32)
        constexpr DWORD BufferSize = 16;
        std::string Buffer(BufferSize, '\0');
        const DWORD CharCount = GetEnvironmentVariableA(EnvVarName, Buffer.data(), BufferSize);

        if (CharCount == 0 || CharCount >= BufferSize) {
            return false;
        }

        Buffer.resize(CharCount);
        return Buffer == "1";
#else
        const char* Value = std::getenv(EnvVarName);
        return Value != nullptr && std::string(Value) == "1";
#endif
    }();

    return Enabled;
}

static bool
MlasConvPyTorchIm2ColEnabledByEnv()
/*++

Routine Description:

    This routine checks whether the experimental PyTorch-like full Im2Col
    convolution path is enabled via environment variable.

Return Value:

    Returns true if the path is enabled.

--*/
{
    static const bool Enabled = []() {
        constexpr const char* EnvVarName = "ORT_MLAS_CONV_PYTORCH_IM2COL_ENABLE";

#if defined(_WIN32)
        constexpr DWORD BufferSize = 16;
        std::string Buffer(BufferSize, '\0');
        const DWORD CharCount = GetEnvironmentVariableA(EnvVarName, Buffer.data(), BufferSize);

        if (CharCount == 0 || CharCount >= BufferSize) {
            return false;
        }

        Buffer.resize(CharCount);
        return Buffer == "1";
#else
        const char* Value = std::getenv(EnvVarName);
        return Value != nullptr && std::string(Value) == "1";
#endif
    }();

    return Enabled;
}

#ifdef _MSC_VER
using MLAS_CONV_ALIGNED_BUFFER = std::unique_ptr<uint8_t, decltype(&_aligned_free)>;
#else
using MLAS_CONV_ALIGNED_BUFFER = std::unique_ptr<uint8_t, decltype(&free)>;
#endif

static MLAS_CONV_ALIGNED_BUFFER
MlasAllocateAlignedBuffer(
    size_t Size
    )
/*++

Routine Description:

    This routine allocates a temporary buffer aligned to the preferred MLAS
    buffer alignment.

Arguments:

    Size - Supplies the number of bytes to allocate.

Return Value:

    Returns the allocated buffer.

--*/
{
    const size_t Alignment = MlasGetPreferredBufferAlignment();

#ifdef _MSC_VER
    MLAS_CONV_ALIGNED_BUFFER Buffer(
        reinterpret_cast<uint8_t*>(_aligned_malloc(Size, Alignment)),
        &_aligned_free);
#elif (__STDC_VERSION__ >= 201112L) && !defined(__APPLE__)
    const size_t AlignedSize = (Size + Alignment - 1) & ~(Alignment - 1);
    MLAS_CONV_ALIGNED_BUFFER Buffer(
        reinterpret_cast<uint8_t*>(aligned_alloc(Alignment, AlignedSize)),
        &free);
#else
    void* Allocation;
    int ErrorCode = posix_memalign(&Allocation, Alignment, Size);
    if (ErrorCode != 0) {
        Allocation = nullptr;
    }

    MLAS_CONV_ALIGNED_BUFFER Buffer(reinterpret_cast<uint8_t*>(Allocation), &free);
#endif

    if (Buffer == nullptr) {
        throw std::bad_alloc();
    }

    return Buffer;
}

struct MLAS_CONV_PACKED_FILTER_CACHE_KEY {
    const float* Filter;
    size_t FilterCount;
    size_t K;
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* BackendKernelSelectorConfig;

    bool operator==(const MLAS_CONV_PACKED_FILTER_CACHE_KEY& Other) const
    {
        return Filter == Other.Filter &&
            FilterCount == Other.FilterCount &&
            K == Other.K &&
            BackendKernelSelectorConfig == Other.BackendKernelSelectorConfig;
    }
};

struct MLAS_CONV_PACKED_FILTER_CACHE_KEY_HASH {
    size_t operator()(const MLAS_CONV_PACKED_FILTER_CACHE_KEY& Key) const noexcept
    {
        size_t HashValue = std::hash<const void*>{}(Key.Filter);
        HashValue ^= std::hash<size_t>{}(Key.FilterCount) + 0x9e3779b9 + (HashValue << 6) + (HashValue >> 2);
        HashValue ^= std::hash<size_t>{}(Key.K) + 0x9e3779b9 + (HashValue << 6) + (HashValue >> 2);
        HashValue ^= std::hash<const void*>{}(Key.BackendKernelSelectorConfig) + 0x9e3779b9 + (HashValue << 6) + (HashValue >> 2);
        return HashValue;
    }
};

#ifdef _MSC_VER
using MLAS_CONV_SHARED_ALIGNED_BUFFER = std::shared_ptr<uint8_t>;

static MLAS_CONV_SHARED_ALIGNED_BUFFER
MlasMakeSharedAlignedBuffer(
    size_t Size
    )
{
    MLAS_CONV_ALIGNED_BUFFER Buffer = MlasAllocateAlignedBuffer(Size);
    return MLAS_CONV_SHARED_ALIGNED_BUFFER(Buffer.release(), &_aligned_free);
}
#else
using MLAS_CONV_SHARED_ALIGNED_BUFFER = std::shared_ptr<uint8_t>;

static MLAS_CONV_SHARED_ALIGNED_BUFFER
MlasMakeSharedAlignedBuffer(
    size_t Size
    )
{
    MLAS_CONV_ALIGNED_BUFFER Buffer = MlasAllocateAlignedBuffer(Size);
    return MLAS_CONV_SHARED_ALIGNED_BUFFER(Buffer.release(), &free);
}
#endif

static const void*
MlasConvGetPackedFilterFromCache(
    const float* Filter,
    size_t FilterCount,
    size_t K,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* BackendKernelSelectorConfig
    )
/*++

Routine Description:

    This routine returns a persistently cached packed GEMM-B filter buffer for
    the supplied convolution filter pointer.

Arguments:

    Filter - Supplies the filter pointer.

    FilterCount - Supplies the number of output channels.

    K - Supplies the flattened per-filter size.

    BackendKernelSelectorConfig - Supplies the backend selector configuration.

Return Value:

    Returns the packed filter buffer.

--*/
{
    static std::mutex CacheMutex;
    static std::unordered_map<MLAS_CONV_PACKED_FILTER_CACHE_KEY,
        MLAS_CONV_SHARED_ALIGNED_BUFFER,
        MLAS_CONV_PACKED_FILTER_CACHE_KEY_HASH> Cache;

    const MLAS_CONV_PACKED_FILTER_CACHE_KEY Key{Filter, FilterCount, K, BackendKernelSelectorConfig};

    std::lock_guard<std::mutex> Guard(CacheMutex);

    const auto Existing = Cache.find(Key);
    if (Existing != Cache.end()) {
        return Existing->second.get();
    }

    const size_t PackedFilterBytes =
        MlasGemmPackBSize(CblasNoTrans, CblasTrans, FilterCount, K, BackendKernelSelectorConfig);

    MLAS_CONV_SHARED_ALIGNED_BUFFER PackedFilterBuffer = MlasMakeSharedAlignedBuffer(PackedFilterBytes);

    MlasGemmPackB(CblasNoTrans, CblasTrans, FilterCount, K, Filter, K,
        PackedFilterBuffer.get(), BackendKernelSelectorConfig);

    const void* PackedFilter = PackedFilterBuffer.get();
    Cache.emplace(Key, std::move(PackedFilterBuffer));

    return PackedFilter;
}

static bool
MlasConvUsePackedSgemmBPath(
    const MLAS_CONV_PARAMETERS* Parameters
    )
/*++

Routine Description:

    This routine determines whether the convolution should bypass the raw
    Im2Col buffer and instead write directly into a packed matrix B buffer
    that can be consumed by the SGEMM packed-B path.

Arguments:

    Parameters - Supplies the structure that contains the convolution
        parameters.

Return Value:

    Returns true if the packed-B fast path should be used.

--*/
{
    MLAS_UNREFERENCED_PARAMETER(Parameters);

    return false;
}

static bool
MlasConvUsePyTorchIm2ColPath(
    const MLAS_CONV_PARAMETERS* Parameters
    )
/*++

Routine Description:

    This routine determines whether the convolution should use the experimental
    full Im2Col path that mirrors PyTorch Slow2d channels-last materialization
    and computes a single large GEMM per sample.

Arguments:

    Parameters - Supplies the structure that contains the convolution
        parameters.

Return Value:

    Returns true if the path should be used.

--*/
{
    return MlasConvPyTorchIm2ColEnabledByEnv() &&
        Parameters->Dimensions == 2 &&
        Parameters->GroupCount == 1 &&
        Parameters->KernelShape[0] == 7 &&
        Parameters->KernelShape[1] == 7 &&
        Parameters->DilationShape[0] == 1 &&
        Parameters->DilationShape[1] == 1 &&
        Parameters->StrideShape[0] == 2 &&
        Parameters->StrideShape[1] == 2 &&
        Parameters->Padding[0] == 3 &&
        Parameters->Padding[1] == 3 &&
        Parameters->Padding[2] == 3 &&
        Parameters->Padding[3] == 3;
}

static void
MlasConvIm2ColPyTorchChannelsLast(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    float* PatchMatrix
    )
/*++

Routine Description:

    This routine materializes the full 2D Im2Col activation matrix using the
    same logical ordering as PyTorch Slow2d with channels-last layout:
    [OutputSize, KernelHeight * KernelWidth * InputChannels].

Arguments:

    Parameters - Supplies the structure that contains the convolution
        parameters.

    Input - Supplies the input tensor.

    PatchMatrix - Supplies the buffer to receive the materialized patch
        matrix.

Return Value:

    None.

--*/
{
    constexpr size_t HeightShapeIndex = 0;
    constexpr size_t WidthShapeIndex = 1;

    const size_t InputHeight = Parameters->InputShape[HeightShapeIndex];
    const size_t InputWidth = Parameters->InputShape[WidthShapeIndex];
    const size_t InputSize = Parameters->InputSize;
    const size_t InputChannels = Parameters->InputChannels;

    const size_t OutputHeight = Parameters->OutputShape[HeightShapeIndex];
    const size_t OutputWidth = Parameters->OutputShape[WidthShapeIndex];

    const size_t KernelHeight = Parameters->KernelShape[HeightShapeIndex];
    const size_t KernelWidth = Parameters->KernelShape[WidthShapeIndex];

    const size_t DilationHeight = Parameters->DilationShape[HeightShapeIndex];
    const size_t DilationWidth = Parameters->DilationShape[WidthShapeIndex];

    const size_t StrideHeight = Parameters->StrideShape[HeightShapeIndex];
    const size_t StrideWidth = Parameters->StrideShape[WidthShapeIndex];

    const ptrdiff_t PaddingTop = ptrdiff_t(Parameters->Padding[HeightShapeIndex]);
    const ptrdiff_t PaddingLeft = ptrdiff_t(Parameters->Padding[WidthShapeIndex]);

    for (size_t oh = 0; oh < OutputHeight; oh++) {
        const ptrdiff_t BaseInputY = ptrdiff_t(oh * StrideHeight) - PaddingTop;

        for (size_t ow = 0; ow < OutputWidth; ow++) {
            const ptrdiff_t BaseInputX = ptrdiff_t(ow * StrideWidth) - PaddingLeft;
            float* OutputRow = PatchMatrix + (oh * OutputWidth + ow) * Parameters->K;

            for (size_t ky = 0; ky < KernelHeight; ky++) {
                const ptrdiff_t InputY = BaseInputY + ptrdiff_t(ky * DilationHeight);

                for (size_t kx = 0; kx < KernelWidth; kx++) {
                    const ptrdiff_t InputX = BaseInputX + ptrdiff_t(kx * DilationWidth);

                    if (InputY < 0 || InputY >= ptrdiff_t(InputHeight) ||
                        InputX < 0 || InputX >= ptrdiff_t(InputWidth)) {
                        std::fill_n(OutputRow, InputChannels, 0.0f);
                        OutputRow += InputChannels;
                        continue;
                    }

                    const size_t InputOffset = size_t(InputY) * InputWidth + size_t(InputX);

                    for (size_t c = 0; c < InputChannels; c++) {
                        *OutputRow++ = Input[c * InputSize + InputOffset];
                    }
                }
            }
        }
    }
}

static void
MlasConvTransposeOutputOToF(
    size_t OutputSize,
    size_t FilterCount,
    float Beta,
    const float* GemmOutput,
    float* Output
    )
/*++

Routine Description:

    This routine transposes a GEMM output buffer of shape [OutputSize,
    FilterCount] into the MLAS convolution output layout [FilterCount,
    OutputSize], applying beta to any existing output values.

Arguments:

    OutputSize - Supplies the logical output size per filter.

    FilterCount - Supplies the number of output channels.

    Beta - Supplies the scalar beta multiplier for the existing output.

    GemmOutput - Supplies the GEMM output buffer in [OutputSize, FilterCount]
        order.

    Output - Supplies the destination convolution output buffer in
        [FilterCount, OutputSize] order.

Return Value:

    None.

--*/
{
    for (size_t f = 0; f < FilterCount; f++) {
        float* OutputRow = Output + f * OutputSize;

        for (size_t o = 0; o < OutputSize; o++) {
            const float Value = GemmOutput[o * FilterCount + f];

            if (Beta == 0.0f) {
                OutputRow[o] = Value;
            } else {
                OutputRow[o] = OutputRow[o] * Beta + Value;
            }
        }
    }
}

static bool
MlasConvUsePackedSgemmBTileTuning(
    const MLAS_CONV_PARAMETERS* Parameters
    )
/*++

Routine Description:

    This routine determines whether the convolution matches the exact MobileClip
    7x7 stride-2 pad-3 geometry used for direct packed-B tile tuning.

Arguments:

    Parameters - Supplies the structure that contains the convolution
        parameters.

Return Value:

    Returns true if tile tuning should be considered.

--*/
{
    return MlasConvTileTuningEnabledByEnv() &&
        Parameters->Dimensions == 2 &&
        Parameters->GroupCount == 1 &&
        Parameters->KernelShape[0] == 7 &&
        Parameters->KernelShape[1] == 7 &&
        Parameters->DilationShape[0] == 1 &&
        Parameters->DilationShape[1] == 1 &&
        Parameters->StrideShape[0] == 2 &&
        Parameters->StrideShape[1] == 2 &&
        Parameters->Padding[0] == 3 &&
        Parameters->Padding[1] == 3 &&
        Parameters->Padding[2] == 3 &&
        Parameters->Padding[3] == 3;
}

static bool
MlasConvUsePackedSgemmBTileTuning32x32(
    const MLAS_CONV_PARAMETERS* Parameters
    )
/*++

Routine Description:

    This routine determines whether the convolution matches the 128->256
    32x32 MobileClip projection that is still regressing with the generic
    direct packed-B path.

Arguments:

    Parameters - Supplies the structure that contains the convolution
        parameters.

Return Value:

    Returns true if the tuned 32x32 tile overrides should be used.

--*/
{
    return MlasConvUsePackedSgemmBTileTuning(Parameters) &&
        Parameters->InputChannels == 128 &&
        Parameters->FilterCount == 256 &&
        Parameters->InputShape[0] == 32 &&
        Parameters->InputShape[1] == 32 &&
        Parameters->OutputShape[0] == 16 &&
        Parameters->OutputShape[1] == 16;
}

static void
MlasConvAdjustPackedSgemmBTiles(
    const MLAS_CONV_PARAMETERS* Parameters,
    size_t SegmentCountN,
    uint32_t* StrideN,
    uint32_t* StrideK
    )
/*++

Routine Description:

    This routine applies env-gated tile overrides for the direct packed-B
    convolution path.

Arguments:

    Parameters - Supplies the structure that contains the convolution
        parameters.

    SegmentCountN - Supplies the current local N segment size.

    StrideN - Supplies the computed N stride to adjust.

    StrideK - Supplies the computed K stride to adjust.

Return Value:

    None.

--*/
{
    if (!MlasConvUsePackedSgemmBTileTuning32x32(Parameters)) {
        return;
    }

    if (SegmentCountN >= 32) {
        *StrideN = 32;
        *StrideK = 256;
    }
}

static inline float*
MlasConvPackedBOutputPointer(
    float* PackedRowBase,
    size_t PackedChunkStride,
    size_t OutputIndex
    )
/*++

Routine Description:

    This routine computes the destination pointer for a packed-B output element
    in the direct Im2Col writer.

Arguments:

    PackedRowBase - Supplies the base address for the current packed K row.

    PackedChunkStride - Supplies the stride between adjacent N slabs for the
        current packed K chunk.

    OutputIndex - Supplies the logical output index within the current N tile.

Return Value:

    Returns the address of the packed-B output element.

--*/
{
    return PackedRowBase +
        (OutputIndex / MLAS_SGEMM_STRIDEN_THREAD_ALIGN) * PackedChunkStride +
        (OutputIndex & (MLAS_SGEMM_STRIDEN_THREAD_ALIGN - 1));
}

void
MlasConvIm2ColToPackedB(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    float* PackedB,
    size_t AlignedCountN,
    size_t k,
    size_t CountK,
    size_t n,
    size_t CountN
    )
/*++

Routine Description:

    This routine converts the input image to a set of convolution patches and
    writes them directly to a packed matrix B buffer compatible with the SGEMM
    packed-B path.

Arguments:

    Parameters - Supplies the structure that contains the convolution
        parameters.

    Input - Supplies the input tensor.

    PackedB - Supplies the buffer to receive the packed convolution patches.

    AlignedCountN - Supplies the padded leading dimension of the packed matrix
        B buffer.

    k - Supplies the K to begin sampling the convolution patches.

    CountK - Supplies the count of K to sample for the convolution patches.

    n - Supplies the N to begin sampling the convolution patches.

    CountN - Supplies the count of N to sample for the convolution patches.

Return Value:

    None.

--*/
{
    constexpr size_t HeightShapeIndex = 0;
    constexpr size_t WidthShapeIndex = 1;
    constexpr size_t PackedColumnCount = MLAS_SGEMM_STRIDEN_THREAD_ALIGN;
    constexpr size_t PackedStrideK = MLAS_SGEMM_PACKED_STRIDEK;

    const size_t OutputWidth = Parameters->OutputShape[WidthShapeIndex];

    const size_t StrideHeight = Parameters->StrideShape[HeightShapeIndex];
    const size_t StrideWidth = Parameters->StrideShape[WidthShapeIndex];

    const size_t nx = (n % OutputWidth);
    const size_t ny = (n / OutputWidth);

    const size_t OriginInputX = nx * StrideWidth;
    const size_t OriginInputY = ny * StrideHeight;

    size_t OutputCountX = OutputWidth - nx;

    const size_t InputHeight = Parameters->InputShape[HeightShapeIndex];
    const size_t InputWidth = Parameters->InputShape[WidthShapeIndex];
    const size_t InputSize = Parameters->InputSize;

    const size_t KernelHeight = Parameters->KernelShape[HeightShapeIndex];
    const size_t KernelWidth = Parameters->KernelShape[WidthShapeIndex];

    size_t kx = (k % KernelWidth);
    size_t ky = (k / KernelWidth) % KernelHeight;

    Input = Input + (k / (KernelHeight * KernelWidth)) * InputSize;

    const size_t DilationHeight = Parameters->DilationShape[HeightShapeIndex];
    const size_t DilationWidth = Parameters->DilationShape[WidthShapeIndex];

    const size_t PaddingLeftY = Parameters->Padding[HeightShapeIndex];
    const size_t PaddingLeftX = Parameters->Padding[WidthShapeIndex];

    std::fill_n(PackedB, AlignedCountN * CountK, 0.0f);

    for (size_t EndingK = k + CountK, RowIndex = 0; k < EndingK; k++, RowIndex++) {

        const size_t PackedChunkStartK = (RowIndex / PackedStrideK) * PackedStrideK;
        const size_t PackedChunkCountK = std::min(CountK - PackedChunkStartK, PackedStrideK);
        const size_t PackedChunkRowIndex = RowIndex - PackedChunkStartK;
        const size_t PackedChunkStride = PackedChunkCountK * PackedColumnCount;

        float* PackedRowBase =
            PackedB + PackedChunkStartK * AlignedCountN + PackedChunkRowIndex * PackedColumnCount;

        size_t CountX = OutputCountX;
        size_t InputY = (ky * DilationHeight) + OriginInputY - PaddingLeftY;
        const size_t RowInitialInputX = (kx * DilationWidth) - PaddingLeftX;
        size_t InitialInputX = RowInitialInputX + OriginInputX;
        size_t RemainingN = CountN;
        size_t OutputIndex = 0;

        do {

            size_t CountXThisRow = CountX;

            if (CountXThisRow > RemainingN) {
                CountXThisRow = RemainingN;
            }

            RemainingN -= CountXThisRow;

            if (InputY < InputHeight) {

                size_t InputX = InitialInputX;
                const float* InputRow = &Input[InputY * InputWidth];

                while (CountXThisRow > 0 && (OutputIndex & (PackedColumnCount - 1)) != 0) {

                    float* PackedScalarOutput =
                        MlasConvPackedBOutputPointer(PackedRowBase, PackedChunkStride, OutputIndex);

                    *PackedScalarOutput = (InputX < InputWidth) ? InputRow[InputX] : 0.0f;
                    InputX += StrideWidth;
                    CountXThisRow--;
                    OutputIndex++;
                }

                while (CountXThisRow >= PackedColumnCount) {

                    float* PackedOutputBlock =
                        MlasConvPackedBOutputPointer(PackedRowBase, PackedChunkStride, OutputIndex);

                    if (InputX >= InputWidth) {
                        break;
                    }

                    const size_t InputXLast = InputX + (PackedColumnCount - 1) * StrideWidth;

                    if (InputXLast >= InputWidth) {
                        break;
                    }

                    if (StrideWidth == 1) {

                        MlasStoreFloat32x4(PackedOutputBlock, MlasLoadFloat32x4(&InputRow[InputX]));
                        MlasStoreFloat32x4(PackedOutputBlock + 4, MlasLoadFloat32x4(&InputRow[InputX + 4]));
                        MlasStoreFloat32x4(PackedOutputBlock + 8, MlasLoadFloat32x4(&InputRow[InputX + 8]));
                        MlasStoreFloat32x4(PackedOutputBlock + 12, MlasLoadFloat32x4(&InputRow[InputX + 12]));

                    } else if (StrideWidth == 2) {

                        PackedOutputBlock[0] = InputRow[InputX];
                        PackedOutputBlock[1] = InputRow[InputX + 2];
                        PackedOutputBlock[2] = InputRow[InputX + 4];
                        PackedOutputBlock[3] = InputRow[InputX + 6];
                        PackedOutputBlock[4] = InputRow[InputX + 8];
                        PackedOutputBlock[5] = InputRow[InputX + 10];
                        PackedOutputBlock[6] = InputRow[InputX + 12];
                        PackedOutputBlock[7] = InputRow[InputX + 14];
                        PackedOutputBlock[8] = InputRow[InputX + 16];
                        PackedOutputBlock[9] = InputRow[InputX + 18];
                        PackedOutputBlock[10] = InputRow[InputX + 20];
                        PackedOutputBlock[11] = InputRow[InputX + 22];
                        PackedOutputBlock[12] = InputRow[InputX + 24];
                        PackedOutputBlock[13] = InputRow[InputX + 26];
                        PackedOutputBlock[14] = InputRow[InputX + 28];
                        PackedOutputBlock[15] = InputRow[InputX + 30];

                    } else {

                        for (size_t Lane = 0; Lane < PackedColumnCount; Lane++) {
                            PackedOutputBlock[Lane] = InputRow[InputX + Lane * StrideWidth];
                        }
                    }

                    InputX += PackedColumnCount * StrideWidth;
                    CountXThisRow -= PackedColumnCount;
                    OutputIndex += PackedColumnCount;
                }

                while (CountXThisRow > 0) {

                    float* PackedOutput =
                        MlasConvPackedBOutputPointer(PackedRowBase, PackedChunkStride, OutputIndex);

                    if (InputX >= InputWidth) {

                        *PackedOutput = 0.0f;
                        InputX += StrideWidth;
                        CountXThisRow--;
                        OutputIndex++;

                    } else if (StrideWidth == 1) {

                        size_t CountCopyX = InputWidth - InputX;

                        if (CountCopyX > CountXThisRow) {
                            CountCopyX = CountXThisRow;
                        }

                        CountXThisRow -= CountCopyX;

                        while (CountCopyX > 0) {
                            float* PackedScalarOutput =
                                MlasConvPackedBOutputPointer(PackedRowBase, PackedChunkStride, OutputIndex);
                            *PackedScalarOutput = InputRow[InputX++];
                            CountCopyX--;
                            OutputIndex++;
                        }

                    } else if (InputX + CountXThisRow * StrideWidth <= InputWidth) {

                        do {
                            float* PackedScalarOutput =
                                MlasConvPackedBOutputPointer(PackedRowBase, PackedChunkStride, OutputIndex);
                            *PackedScalarOutput = InputRow[InputX];
                            InputX += StrideWidth;
                            OutputIndex++;
                        } while (--CountXThisRow > 0);

                    } else {

                        do {
                            float* PackedScalarOutput =
                                MlasConvPackedBOutputPointer(PackedRowBase, PackedChunkStride, OutputIndex);
                            *PackedScalarOutput = (InputX < InputWidth) ? InputRow[InputX] : 0.0f;
                            InputX += StrideWidth;
                            OutputIndex++;
                        } while (--CountXThisRow > 0);
                    }
                }

            } else {

                OutputIndex += CountXThisRow;
            }

            CountX = OutputWidth;
            InputY += StrideHeight;
            InitialInputX = RowInitialInputX;

        } while (RemainingN > 0);

        if (++kx == KernelWidth) {

            if (++ky == KernelHeight) {

                Input += InputSize;

                ky = 0;
            }

            kx = 0;
        }
    }
}

void
MlasConvVol2Col(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    float* ColumnBuffer,
    size_t k,
    size_t CountK,
    size_t n,
    size_t CountN
    )
/*++

Routine Description:

    This routine converts the input volume to a set of convolution patches
    appropriate for use with a GEMM operation.

    This implementation supports sampling a portion of the convolution
    patches. This avoids the need to allocate very large buffers to store
    all of the convolution patches at once, when the underlying GEMM
    implementation will already break up the operation into panels. Multiple
    threads can also be used to process different portions of the image.

Arguments:

    Parameters - Supplies the structure that contains the convolution
        parameters.

    Input - Supplies the input tensor.

    ColumnBuffer - Supplies the buffer to receive the convolution patches.

    k - Supplies the K to begin sampling the convolution patches.

    CountK - Supplies the count of K to sample for the convolution patches.

    n - Supplies the N to begin sampling the convolution patches.

    CountN - Supplies the count of N to sample for the convolution patches.

Return Value:

    None.

--*/
{
    constexpr size_t DepthShapeIndex = 0;
    constexpr size_t HeightShapeIndex = 1;
    constexpr size_t WidthShapeIndex = 2;

    const size_t OutputHeight = Parameters->OutputShape[HeightShapeIndex];
    const size_t OutputWidth = Parameters->OutputShape[WidthShapeIndex];

    const size_t StrideDepth = Parameters->StrideShape[DepthShapeIndex];
    const size_t StrideHeight = Parameters->StrideShape[HeightShapeIndex];
    const size_t StrideWidth = Parameters->StrideShape[WidthShapeIndex];

    const size_t nx = (n % OutputWidth);
    const size_t ny = ((n / OutputWidth) % OutputHeight);
    const size_t nz = ((n / OutputWidth) / OutputHeight);

    size_t OutputCountX = OutputWidth - nx;
    size_t OutputCountY = OutputHeight - ny;

    const size_t OriginInputX = nx * StrideWidth;
    const size_t OriginInputY = ny * StrideHeight;
    const size_t OriginInputZ = nz * StrideDepth;

    const size_t InputDepth = Parameters->InputShape[DepthShapeIndex];
    const size_t InputHeight = Parameters->InputShape[HeightShapeIndex];
    const size_t InputWidth = Parameters->InputShape[WidthShapeIndex];
    const size_t InputSize = Parameters->InputSize;

    const size_t KernelDepth = Parameters->KernelShape[DepthShapeIndex];
    const size_t KernelHeight = Parameters->KernelShape[HeightShapeIndex];
    const size_t KernelWidth = Parameters->KernelShape[WidthShapeIndex];

    size_t kx = (k % KernelWidth);
    size_t ky = (k / KernelWidth) % KernelHeight;
    size_t kz = ((k / KernelWidth) / KernelHeight) % KernelDepth;

    Input = Input + (k / (KernelDepth * KernelHeight * KernelWidth)) * InputSize;

    const size_t DilationDepth = Parameters->DilationShape[DepthShapeIndex];
    const size_t DilationHeight = Parameters->DilationShape[HeightShapeIndex];
    const size_t DilationWidth = Parameters->DilationShape[WidthShapeIndex];

    const size_t PaddingLeftZ = Parameters->Padding[DepthShapeIndex];
    const size_t PaddingLeftY = Parameters->Padding[HeightShapeIndex];
    const size_t PaddingLeftX = Parameters->Padding[WidthShapeIndex];

    for (size_t EndingK = k + CountK; k < EndingK; k++) {

        size_t CountY = OutputCountY;
        size_t CountX = OutputCountX;
        size_t InputZ = (kz * DilationDepth) + OriginInputZ - PaddingLeftZ;
        const size_t RowInitialInputY = (ky * DilationHeight) - PaddingLeftY;
        size_t InputY = RowInitialInputY + OriginInputY;
        const size_t RowInitialInputX = (kx * DilationWidth) - PaddingLeftX;
        size_t InitialInputX = RowInitialInputX + OriginInputX;
        size_t RemainingN = CountN;

        do {

            if (CountX > RemainingN) {
                CountX = RemainingN;
            }

            RemainingN -= CountX;

            //
            // Check if the input is in the top/bottom or front/back padding region.
            //

            if (InputY < InputHeight && InputZ < InputDepth) {

                size_t InputX = InitialInputX;
                const float* InputRow = &Input[InputZ * (InputHeight * InputWidth) + InputY * InputWidth];

                do {

                    //
                    // Check if the input is in the left/right padding region.
                    //

                    if (InputX >= InputWidth) {

                        *ColumnBuffer++ = 0;
                        InputX += StrideWidth;
                        CountX--;

                    } else if (StrideWidth == 1) {

                        //
                        // Copy input elements to the column buffer.
                        //

                        size_t CountCopyX = InputWidth - InputX;

                        if (CountCopyX > CountX) {
                            CountCopyX = CountX;
                        }

                        CountX -= CountCopyX;

                        while (CountCopyX >= 4) {
                            MlasStoreFloat32x4(ColumnBuffer, MlasLoadFloat32x4(&InputRow[InputX]));
                            ColumnBuffer += 4;
                            InputX += 4;
                            CountCopyX -= 4;
                        }

                        while (CountCopyX > 0) {
                            *ColumnBuffer++ = InputRow[InputX++];
                            CountCopyX--;
                        }

                    } else if (InputX + CountX * StrideWidth <= InputWidth) {

                        do {
                            *ColumnBuffer++ = InputRow[InputX];
                            InputX += StrideWidth;
                        } while (--CountX > 0);

                    } else {

                        do {
                            *ColumnBuffer++ = (InputX < InputWidth) ? InputRow[InputX] : 0;
                            InputX += StrideWidth;
                        } while (--CountX > 0);
                    }

                } while (CountX > 0);

            } else {

                //
                // The entire input row is in the padding region.
                //

                MLAS_FLOAT32X4 ZeroFloat32x4 = MlasZeroFloat32x4();

                while (CountX >= 4) {
                    MlasStoreFloat32x4(ColumnBuffer, ZeroFloat32x4);
                    ColumnBuffer += 4;
                    CountX -= 4;
                }

                while (CountX > 0) {
                    MlasStoreLaneFloat32x4<0>(ColumnBuffer, ZeroFloat32x4);
                    ColumnBuffer++;
                    CountX--;
                }
            }

            CountX = OutputWidth;
            InputY += StrideHeight;
            InitialInputX = RowInitialInputX;

            if (--CountY == 0) {

                InputY = RowInitialInputY;
                InputZ += StrideDepth;

                CountY = OutputHeight;
            }

        } while (RemainingN > 0);

        //
        // Advance the kernel indices and advance to the next channel if the
        // entire kernel is complete.
        //

        if (++kx == KernelWidth) {

            if (++ky == KernelHeight) {

                if (++kz == KernelDepth) {

                    Input += InputSize;

                    kz = 0;
                }

                ky = 0;
            }

            kx = 0;
        }
    }
}

void
MlasConvOperation(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    const float* Filter,
    const float* Bias,
    float* ColumnBuffer,
    float* Output,
    size_t SegmentStartN,
    size_t SegmentCountN
    )
/*++

Routine Description:

    This routine implements the convolution operation.

Arguments:

    Parameters - Supplies the structure that contains the convolution
        parameters.

    Input - Supplies the input tensor.

    Filter - Supplies the filter tensor.

    Bias - Optionally supplies the bias vector.

    ColumnBuffer - Supplies the thread local slice of the working buffer.

    Output - Supplies the output tensor.

    SegmentStartN - Supplies the N to begin sampling the convolution patches.

    SegmentCountN - Supplies the count of N to sample for the convolution
        patches.

Return Value:

    None.

--*/
{
    const size_t FilterCount = Parameters->FilterCount;
    const size_t OutputSize = Parameters->OutputSize;
    const size_t K = Parameters->K;
    const bool UsePackedSgemmBPath = MlasConvUsePackedSgemmBPath(Parameters);

    //
    // Compute the strides to step through slices of the local segment.
    //
    // See MlasSgemmOperation.
    //

    uint32_t StrideN = MLAS_SGEMM_STRIDEN;
    uint32_t StrideK = MLAS_SGEMM_STRIDEK;

    if (SegmentCountN >= K) {

        while (StrideK / 2 >= K) {
            StrideN *= 2;
            StrideK /= 2;
        }

    } else {

        while (StrideN > 16 && StrideN / 2 >= SegmentCountN) {
            StrideK *= 2;
            StrideN /= 2;
        }
    }

    if (UsePackedSgemmBPath) {
        MlasConvAdjustPackedSgemmBTiles(Parameters, SegmentCountN, &StrideN, &StrideK);
    }

    MLAS_CONV_ALIGNED_BUFFER PackedColumnBuffer(nullptr,
    #ifdef _MSC_VER
        &_aligned_free
    #else
        &free
    #endif
        );

    if (UsePackedSgemmBPath) {
        PackedColumnBuffer = MlasAllocateAlignedBuffer(
            MlasGemmPackBSize(CblasNoTrans, CblasNoTrans, StrideN, StrideK, nullptr));
    }

    //
    // Step through each slice of the input tensor along the N dimension.
    //

    size_t CountN;

    for (size_t n = 0; n < SegmentCountN; n += CountN) {

        CountN = SegmentCountN - n;

        if (CountN > StrideN) {
            CountN = StrideN;
        }

        //
        // Step through each slice of the input tensor along the K dimension.
        //

        size_t CountK;
        float beta = Parameters->Beta;
        float* SegmentOutput = Output + SegmentStartN + n;

        for (size_t k = 0; k < K; k += CountK) {

            CountK = K - k;

            if (CountK > StrideK) {
                CountK = StrideK;
            }

            if (UsePackedSgemmBPath) {
                float* PackedColumnBufferData = reinterpret_cast<float*>(PackedColumnBuffer.get());

                const size_t AlignedCountN =
                    (CountN + MLAS_SGEMM_STRIDEN_THREAD_ALIGN - 1) &
                    ~(size_t(MLAS_SGEMM_STRIDEN_THREAD_ALIGN) - 1);

                MlasConvIm2ColToPackedB(Parameters, Input, PackedColumnBufferData,
                    AlignedCountN, k, CountK, SegmentStartN + n, CountN);

                MlasGemm(CblasNoTrans, FilterCount, CountN, CountK, 1.0f,
                    Filter + k, K, PackedColumnBufferData, beta, SegmentOutput,
                    OutputSize, nullptr, nullptr);

            } else if (Parameters->Dimensions == 2) {
                MlasConvIm2Col(Parameters, Input, ColumnBuffer, k, CountK,
                    SegmentStartN + n, CountN);
            } else {
                MlasConvVol2Col(Parameters, Input, ColumnBuffer, k, CountK,
                    SegmentStartN + n, CountN);
            }

            if (!UsePackedSgemmBPath) {
                MlasSgemmOperation(CblasNoTrans, CblasNoTrans, FilterCount, CountN,
                    CountK, 1.0f, Filter + k, K, ColumnBuffer, CountN, beta,
                    SegmentOutput, OutputSize);
            }

            beta = 1.0f;
        }

        //
        // Apply the activation with optional bias.
        //

        MlasActivation(Parameters->Activation, SegmentOutput, Bias, FilterCount,
            CountN, OutputSize);
    }
}

void
MlasConvOperationThreaded(
    void* Context,
    ptrdiff_t Index
    )
/*++

Routine Description:

    This routine is invoked from a worker thread to execute a segment of a
    convolution operation.

Arguments:

    Context - Supplies the pointer to the context for the threaded operation.

    Index - Supplies the current index of the threaded operation.

Return Value:

    None.

--*/
{
    MLAS_CONV_WORK_BLOCK* WorkBlock = (MLAS_CONV_WORK_BLOCK*)Context;

    MLAS_CONV_WORK_BLOCK::SEGMENT* Segment = &WorkBlock->Segments[Index];

    float* ColumnBuffer =
        WorkBlock->WorkingBuffer + Index * MLAS_CONV_WORKING_BUFFER_SIZE_PER_THREAD;

    MlasConvOperation(WorkBlock->Parameters, WorkBlock->Input, WorkBlock->Filter,
        WorkBlock->Bias, ColumnBuffer, WorkBlock->Output, Segment->StartN,
        Segment->CountN);
}

void
MlasConvGemmDirectThreaded(
    void* Context,
    ptrdiff_t Index
    )
/*++

Routine Description:

    This routine is invoked from a worker thread to execute a segment of a
    convolution operation.

Arguments:

    Context - Supplies the pointer to the context for the threaded operation.

    Index - Supplies the current index of the threaded operation.

Return Value:

    None.

--*/
{
    MLAS_CONV_WORK_BLOCK* WorkBlock = (MLAS_CONV_WORK_BLOCK*)Context;

    const MLAS_CONV_PARAMETERS* Parameters = WorkBlock->Parameters;

    //
    // Compute the range of indices to use for this thread.
    //

    const size_t GroupCount = Parameters->GroupCount;
    const size_t BatchGroupCount = Parameters->BatchCount * GroupCount;
    const float Beta = Parameters->Beta;

    size_t BatchGroupStart;
    size_t BatchGroupRemaining;

    MlasPartitionWork(Index, WorkBlock->TargetThreadCount, BatchGroupCount,
        &BatchGroupStart, &BatchGroupRemaining);

    size_t BatchGroupEnd = BatchGroupStart + BatchGroupRemaining;

    //
    // Iterate over the batch and groups allocated to this thread.
    //

    const size_t FilterCount = Parameters->FilterCount;
    const size_t OutputSize = Parameters->OutputSize;
    const size_t K = Parameters->K;

    const size_t InputGroupSize = Parameters->InputChannels * Parameters->InputSize;
    const size_t OutputGroupSize = FilterCount * OutputSize;
    const size_t FilterGroupSize = FilterCount * K;

    const float* input = WorkBlock->Input + BatchGroupStart * InputGroupSize;
    float* output = WorkBlock->Output + BatchGroupStart * OutputGroupSize;

    for (size_t bg = BatchGroupStart; bg < BatchGroupEnd; bg++) {

        size_t group = bg % GroupCount;
        const float* filter = WorkBlock->Filter + group * FilterGroupSize;

        //
        // Invoke the non-threaded GEMM directly with the input tensor.
        //

        MlasSgemmOperation(CblasNoTrans, Parameters->u.GemmDirect.TransB, FilterCount, OutputSize,
                           K, 1.0f, filter, K, input, Parameters->u.GemmDirect.ldb, Beta, output,
                           OutputSize);

        //
        // Apply the activation with optional bias.
        //

        const float* bias = WorkBlock->Bias;

        if (bias != nullptr) {
            bias += group * FilterCount;
        }

        MlasActivation(Parameters->Activation, output, bias, FilterCount,
            OutputSize, OutputSize);

        input += InputGroupSize;
        output += OutputGroupSize;
    }
}

void
MlasConvExpandThenGemmSegmentedThreaded(
    void* Context,
    ptrdiff_t Index
)
/*++

Routine Description:

    This routine is invoked from a worker thread to execute a segment of a
    convolution operation.

    If using this, the entire convolution operation is parallelized on the
    (batch size * group count) parameter and this routine has logic to
    perform a specific thread's shard of the entire Convolution operation.

Arguments:

    Context - Supplies the pointer to the context for the threaded operation.

    Index - Supplies the current index of the threaded operation.

Return Value:

    None.

--*/

{
    MLAS_CONV_WORK_BLOCK* WorkBlock = (MLAS_CONV_WORK_BLOCK*)Context;

    const MLAS_CONV_PARAMETERS* Parameters = WorkBlock->Parameters;

    const size_t GroupCount = Parameters->GroupCount;
    const size_t BatchGroupCount = Parameters->BatchCount * GroupCount;

    const size_t TargetThreadCount = WorkBlock->TargetThreadCount;

    const size_t BatchGroupCountPerThread = BatchGroupCount / TargetThreadCount;
    const size_t BatchGroupCountExtra = BatchGroupCount % TargetThreadCount;

    size_t BatchGroupStart;
    size_t BatchGroupEnd;

    if (static_cast<size_t>(Index) < BatchGroupCountExtra) {
        BatchGroupStart = (BatchGroupCountPerThread + 1) * Index;
        BatchGroupEnd = BatchGroupStart + BatchGroupCountPerThread + 1;
    } else {
        BatchGroupStart = BatchGroupCountPerThread * Index + BatchGroupCountExtra;
        BatchGroupEnd = BatchGroupStart + BatchGroupCountPerThread;
    }

    const size_t FilterCount = Parameters->FilterCount;
    const size_t OutputSize = Parameters->OutputSize;
    const size_t K = Parameters->K;

    const size_t InputGroupSize = Parameters->InputChannels * Parameters->InputSize;
    const size_t OutputGroupSize = FilterCount * OutputSize;
    const size_t FilterGroupSize = FilterCount * K;

    for (size_t bg = BatchGroupStart; bg < BatchGroupEnd; bg++) {
        size_t group = bg % GroupCount;

        const float* input = WorkBlock->Input + bg * InputGroupSize;
        const float* filter = WorkBlock->Filter + group * FilterGroupSize;
        float* output = WorkBlock->Output + bg * OutputGroupSize;
        const float* bias = WorkBlock->Bias;
        if (bias != nullptr) {
            bias += group * FilterCount;
        }
        float* ColumnBuffer = WorkBlock->WorkingBuffer + Index * OutputSize * K;

        MlasConvOperation(Parameters, input, filter, bias, ColumnBuffer, output, 0, OutputSize);
    }
}

#if defined(MLAS_TARGET_WASM_SCALAR) || defined(MLAS_TARGET_ARM64)

void
MlasDepthwiseThreaded(
    void* Context,
    ptrdiff_t Index
)

/*++

Routine Description:

    This routine is invoked from a worker thread to execute a segment of a
    convolution operation.

    If using this, the entire convolution operation is parallelized on the
    (batch size * group count) parameter and this routine has logic to
    perform a specific thread's shard of the entire Convolution operation.

Arguments:

    Context - Supplies the pointer to the context for the threaded operation.

    Index - Supplies the current index of the threaded operation.

Return Value:

    None.

--*/

{

    MLAS_CONV_WORK_BLOCK* WorkBlock = (MLAS_CONV_WORK_BLOCK*)Context;

    const MLAS_CONV_PARAMETERS* Parameters = WorkBlock->Parameters;

    const size_t GroupCount = Parameters->GroupCount;
    const size_t BatchGroupCount = Parameters->BatchCount * GroupCount;

    const size_t TargetThreadCount = WorkBlock->TargetThreadCount;

    const size_t BatchGroupCountPerThread = BatchGroupCount / TargetThreadCount;
    const size_t BatchGroupCountExtra = BatchGroupCount % TargetThreadCount;

    size_t BatchGroupStart;
    size_t BatchGroupEnd;

    if (static_cast<size_t>(Index) < BatchGroupCountExtra) {
        BatchGroupStart = (BatchGroupCountPerThread + 1) * Index;
        BatchGroupEnd = BatchGroupStart + BatchGroupCountPerThread + 1;
    } else {
        BatchGroupStart = BatchGroupCountPerThread * Index + BatchGroupCountExtra;
        BatchGroupEnd = BatchGroupStart + BatchGroupCountPerThread;
    }

    const size_t FilterCount = Parameters->FilterCount;
    const size_t OutputSize = Parameters->OutputSize;
    const size_t K = Parameters->K;

    const size_t InputGroupSize = Parameters->InputChannels * Parameters->InputSize;
    const size_t OutputGroupSize = FilterCount * OutputSize;
    const size_t FilterGroupSize = FilterCount * K;

    for (size_t bg = BatchGroupStart; bg < BatchGroupEnd; bg++) {
        size_t group = bg % GroupCount;

        const float* input = WorkBlock->Input + bg * InputGroupSize;
        const float* filter = WorkBlock->Filter + group * FilterGroupSize;
        float* output = WorkBlock->Output + bg * OutputGroupSize;
        const float* bias = WorkBlock->Bias;
        if (bias != nullptr) {
            bias += group * FilterCount;
        }

        float* WorkingBuffer = WorkBlock->WorkingBuffer;

        MlasConvDepthwiseFloat_CHW(Parameters, input, filter, output, WorkingBuffer);
        MlasActivation(Parameters->Activation, output, bias, FilterCount, OutputSize, OutputSize);
    }
}

#endif

void
MlasDepthwiseMultiplier2Threaded(
    void* Context,
    ptrdiff_t Index
)
/*++

Routine Description:

    This routine is invoked from a worker thread to execute a segment of a
    grouped depthwise-with-multiplier-2 convolution operation.

Arguments:

    Context - Supplies the pointer to the context for the threaded operation.

    Index - Supplies the current index of the threaded operation.

Return Value:

    None.

--*/
{
    MLAS_CONV_WORK_BLOCK* WorkBlock = (MLAS_CONV_WORK_BLOCK*)Context;

    const MLAS_CONV_PARAMETERS* Parameters = WorkBlock->Parameters;

    const size_t GroupCount = Parameters->GroupCount;
    const size_t BatchGroupCount = Parameters->BatchCount * GroupCount;

    const size_t TargetThreadCount = WorkBlock->TargetThreadCount;

    const size_t BatchGroupCountPerThread = BatchGroupCount / TargetThreadCount;
    const size_t BatchGroupCountExtra = BatchGroupCount % TargetThreadCount;

    size_t BatchGroupStart;
    size_t BatchGroupEnd;

    if (static_cast<size_t>(Index) < BatchGroupCountExtra) {
        BatchGroupStart = (BatchGroupCountPerThread + 1) * Index;
        BatchGroupEnd = BatchGroupStart + BatchGroupCountPerThread + 1;
    } else {
        BatchGroupStart = BatchGroupCountPerThread * Index + BatchGroupCountExtra;
        BatchGroupEnd = BatchGroupStart + BatchGroupCountPerThread;
    }

    const size_t FilterCount = Parameters->FilterCount;
    const size_t OutputSize = Parameters->OutputSize;
    const size_t K = Parameters->K;

    const size_t InputGroupSize = Parameters->InputChannels * Parameters->InputSize;
    const size_t OutputGroupSize = FilterCount * OutputSize;
    const size_t FilterGroupSize = FilterCount * K;

    for (size_t bg = BatchGroupStart; bg < BatchGroupEnd; bg++) {
        const size_t group = bg % GroupCount;

        const float* input = WorkBlock->Input + bg * InputGroupSize;
        const float* filter = WorkBlock->Filter + group * FilterGroupSize;
        float* output = WorkBlock->Output + bg * OutputGroupSize;
        const float* bias = WorkBlock->Bias;
        if (bias != nullptr) {
            bias += group * FilterCount;
        }

        MlasConvDepthwiseMultiplier2FloatCHW(Parameters, input, filter, output);
        MlasActivation(Parameters->Activation, output, bias, FilterCount, OutputSize, OutputSize);
    }
}

inline
bool
MlasConvTryMultithread(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    const float* Filter,
    const float* Bias,
    float* WorkingBuffer,
    float* Output,
    MLAS_THREADPOOL* ThreadPool
    )
/*++

Routine Description:

    This routine attempts to launch a convolution operation across multiple
    threads.

Arguments:

    Parameters - Supplies the structure that contains the convolution
        parameters.

    Input - Supplies the input tensor.

    Filter - Supplies the filter tensor.

    Bias - Optionally supplies the bias vector.

    WorkingBuffer - Supplies a working buffer sized to the number of elements
        returned by MlasConvPrepare.

    Output - Supplies the output tensor.

    ThreadPool - Supplies the thread pool object to use, else nullptr if the
        base library threading support should be used.

Return Value:

    Returns true if the operation was completed across multiple threads, else
    false if the operation should fall back to a single thread.

--*/
{
    MLAS_CONV_WORK_BLOCK WorkBlock;

    const size_t OutputSize = Parameters->OutputSize;
    const size_t ThreadStrideN = Parameters->u.ExpandThenGemmSegmented.ThreadStrideN;

    if (ThreadStrideN >= OutputSize) {
        return false;
    }

    //
    // Initialize the common fields of the work block.
    //

    WorkBlock.Parameters = Parameters;
    WorkBlock.Input = Input;
    WorkBlock.Filter = Filter;
    WorkBlock.Bias = Bias;
    WorkBlock.WorkingBuffer = WorkingBuffer;
    WorkBlock.Output = Output;

    //
    // Segment the operation across multiple threads.
    //

    int32_t Index = 0;
    size_t SegmentCountN;

    for (size_t SegmentStartN = 0; SegmentStartN < OutputSize; SegmentStartN += SegmentCountN) {

        SegmentCountN = OutputSize - SegmentStartN;

        if (SegmentCountN > ThreadStrideN) {
            SegmentCountN = ThreadStrideN;
        }

        WorkBlock.Segments[Index].StartN = SegmentStartN;
        WorkBlock.Segments[Index].CountN = SegmentCountN;

        Index++;
    }

    MlasExecuteThreaded(MlasConvOperationThreaded, &WorkBlock, Index, ThreadPool);

    return true;
}

void
MLASCALL
MlasConv(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    const float* Filter,
    const float* Bias,
    float* WorkingBuffer,
    float* Output,
    MLAS_THREADPOOL* ThreadPool
    )
/*++

Routine Description:

    This routine implements the convolution operation.

Arguments:

    Parameters - Supplies the structure that contains the convolution
        parameters.

    Input - Supplies the input tensor.

    Filter - Supplies the filter tensor.

    Bias - Optionally supplies the bias vector.

    WorkingBuffer - Supplies a working buffer sized to the number of elements
        returned by MlasConvPrepare.

    Output - Supplies the output tensor.

    ThreadPool - Supplies the thread pool object to use, else nullptr if the
        base library threading support should be used.

Return Value:

    None.

--*/
{
    // Override
    if(GetMlasPlatform().MlasConvOverride != nullptr &&
        GetMlasPlatform().MlasConvOverride(Parameters,Input,Filter,Bias,WorkingBuffer,Output,ThreadPool)){
    return;
    }

    const size_t FilterCount = Parameters->FilterCount;
    const size_t OutputSize = Parameters->OutputSize;
    const size_t K = Parameters->K;

    const size_t InputGroupSize = Parameters->InputChannels * Parameters->InputSize;
    const size_t OutputGroupSize = FilterCount * OutputSize;
    const size_t FilterGroupSize = FilterCount * K;

    const size_t BatchCount = Parameters->BatchCount;
    const size_t GroupCount = Parameters->GroupCount;

    const MLAS_CONV_ALGORITHM Algorithm = Parameters->Algorithm;

    //
    // Schedule batches of GEMMs across multiple threads.
    //

    if (Algorithm == MlasConvAlgorithmGemmDirect && ((BatchCount > 1) || (GroupCount > 1))) {

        const size_t BatchGroupCount = BatchCount * GroupCount;

        ptrdiff_t TargetThreadCount = MlasGetMaximumThreadCount(ThreadPool);

        if (static_cast<size_t>(TargetThreadCount) >= BatchGroupCount) {
            TargetThreadCount = static_cast<ptrdiff_t>(BatchGroupCount);
        }

        MLAS_CONV_WORK_BLOCK WorkBlock;

        WorkBlock.Parameters = Parameters;
        WorkBlock.Input = Input;
        WorkBlock.Filter = Filter;
        WorkBlock.Bias = Bias;
        WorkBlock.WorkingBuffer = nullptr;
        WorkBlock.Output = Output;
        WorkBlock.TargetThreadCount = TargetThreadCount;

        MlasExecuteThreaded(MlasConvGemmDirectThreaded, &WorkBlock, TargetThreadCount, ThreadPool);

        return;
    }

#if defined(MLAS_TARGET_WASM_SCALAR) || defined(MLAS_TARGET_ARM64)

    if (Algorithm == MlasConvAlgorithmDepthwise) {
        // Fill the Working Buffer with Zero for use by the depthwise kernel.
        // The length for the zeros are input image wide + 2 currently.
        std::fill_n(WorkingBuffer, Parameters->InputShape[1] + 2, 0.0f);
    }

#endif

    if (Algorithm == MlasConvAlgorithmDepthwiseMultiplier2 && ((BatchCount > 1) || (GroupCount > 1))) {
        const size_t BatchGroupCount = BatchCount * GroupCount;

        ptrdiff_t TargetThreadCount = MlasGetMaximumThreadCount(ThreadPool);

        if (static_cast<size_t>(TargetThreadCount) >= BatchGroupCount) {
            TargetThreadCount = static_cast<ptrdiff_t>(BatchGroupCount);
        }

        MLAS_CONV_WORK_BLOCK WorkBlock;

        WorkBlock.Parameters = Parameters;
        WorkBlock.Input = Input;
        WorkBlock.Filter = Filter;
        WorkBlock.Bias = Bias;
        WorkBlock.WorkingBuffer = nullptr;
        WorkBlock.Output = Output;
        WorkBlock.TargetThreadCount = TargetThreadCount;

        MlasExecuteThreaded(MlasDepthwiseMultiplier2Threaded, &WorkBlock, TargetThreadCount, ThreadPool);

        return;
    }

    if (Algorithm == MlasConvAlgorithmExpandThenGemmSegmented && ((BatchCount > 1) || (GroupCount > 1))) {
        const size_t BatchGroupCount = BatchCount * GroupCount;

        ptrdiff_t TargetThreadCount = MlasGetMaximumThreadCount(ThreadPool);

        if (static_cast<size_t>(TargetThreadCount) >= BatchGroupCount) {
            TargetThreadCount = static_cast<ptrdiff_t>(BatchGroupCount);
        }

        MLAS_CONV_WORK_BLOCK WorkBlock;

        WorkBlock.Parameters = Parameters;
        WorkBlock.Input = Input;
        WorkBlock.Filter = Filter;
        WorkBlock.Bias = Bias;
        WorkBlock.WorkingBuffer = WorkingBuffer;
        WorkBlock.Output = Output;
        WorkBlock.TargetThreadCount = TargetThreadCount;

        MlasExecuteThreaded(MlasConvExpandThenGemmSegmentedThreaded, &WorkBlock, TargetThreadCount, ThreadPool);

        return;
    }


#if defined(MLAS_TARGET_WASM_SCALAR) || defined(MLAS_TARGET_ARM64)

    if (Algorithm == MlasConvAlgorithmDepthwise && ((BatchCount > 1) || (GroupCount > 1))) {
        const size_t BatchGroupCount = BatchCount * GroupCount;

        ptrdiff_t TargetThreadCount = MlasGetMaximumThreadCount(ThreadPool);

        if (static_cast<size_t>(TargetThreadCount) >= BatchGroupCount) {
            TargetThreadCount = static_cast<ptrdiff_t>(BatchGroupCount);
        }

        MLAS_CONV_WORK_BLOCK WorkBlock;

        WorkBlock.Parameters = Parameters;
        WorkBlock.Input = Input;
        WorkBlock.Filter = Filter;
        WorkBlock.Bias = Bias;
        WorkBlock.WorkingBuffer = WorkingBuffer;
        WorkBlock.Output = Output;
        WorkBlock.TargetThreadCount = TargetThreadCount;

        MlasExecuteThreaded(MlasDepthwiseThreaded, &WorkBlock, TargetThreadCount, ThreadPool);

        return;
    }

#endif

    //
    // Iterate over each batch and group.
    //
    for (size_t batch = 0; batch < BatchCount; batch++) {

        const float* filter = Filter;
        const float* bias = Bias;

        for (size_t group = 0; group < GroupCount; group++) {

            //
            // Dispatch the convolution.
            //

            switch (Algorithm) {

                case MlasConvAlgorithmGemmDirect:
                {
                    //
                    // Invoke the threaded GEMM directly with the input tensor.
                    //

                    MlasGemm(CblasNoTrans, Parameters->u.GemmDirect.TransB, FilterCount, OutputSize,
                             K, 1.0f, filter, K, Input, Parameters->u.GemmDirect.ldb,
                             Parameters->Beta, Output, OutputSize, ThreadPool, Parameters->BackendKernelSelectorConfig);

                    //
                    // Apply the activation with optional bias.
                    //

                    MlasActivation(Parameters->Activation, Output, bias, FilterCount,
                        OutputSize, OutputSize);

                    break;
                }

                case MlasConvAlgorithmExpandThenGemm:
                {
                    //
                    // Expand the input tensor to the working buffer and then invoke the
                    // threaded GEMM.
                    //

                    if (Parameters->Dimensions == 2) {
                        MlasConvIm2Col(Parameters, Input, WorkingBuffer, 0, K, 0, OutputSize);
                    } else {
                        MlasConvVol2Col(Parameters, Input, WorkingBuffer, 0, K, 0, OutputSize);
                    }

                    MlasGemm(CblasNoTrans, CblasNoTrans, FilterCount, OutputSize, K, 1.0f, filter,
                             K, WorkingBuffer, OutputSize, Parameters->Beta, Output, OutputSize,
                             ThreadPool, Parameters->BackendKernelSelectorConfig);

                    //
                    // Apply the activation with optional bias.
                    //

                    MlasActivation(Parameters->Activation, Output, bias, FilterCount,
                        OutputSize, OutputSize);

                    break;
                }

                case MlasConvAlgorithmExpandThenGemmPyTorchIm2Col:
                {
                    float* PatchMatrix = WorkingBuffer;
                    float* GemmOutputBuffer = PatchMatrix + OutputSize * K;

                    MlasConvIm2ColPyTorchChannelsLast(Parameters, Input, PatchMatrix);
                    const void* PackedFilterBuffer = MlasConvGetPackedFilterFromCache(
                        filter, FilterCount, K, Parameters->BackendKernelSelectorConfig);

                    MlasGemm(CblasNoTrans, OutputSize, FilterCount, K, 1.0f,
                        PatchMatrix, K, PackedFilterBuffer, 0.0f,
                        GemmOutputBuffer, FilterCount, ThreadPool,
                        Parameters->BackendKernelSelectorConfig);

                    MlasConvTransposeOutputOToF(OutputSize, FilterCount, Parameters->Beta,
                        GemmOutputBuffer, Output);

                    MlasActivation(Parameters->Activation, Output, bias, FilterCount,
                        OutputSize, OutputSize);

                    break;
                }

                case MlasConvAlgorithmDepthwiseMultiplier2:
                {
                    MlasConvDepthwiseMultiplier2FloatCHW(Parameters, Input, filter, Output);
                    MlasActivation(Parameters->Activation, Output, bias, FilterCount, OutputSize, OutputSize);
                    break;
                }

#if defined(MLAS_TARGET_WASM_SCALAR) || defined(MLAS_TARGET_ARM64)

                case MlasConvAlgorithmDepthwise:
                {
                    MlasConvDepthwiseFloat_CHW(Parameters, Input, filter, Output, WorkingBuffer);
                    MlasActivation(Parameters->Activation, Output, bias, FilterCount, OutputSize, OutputSize);
                    break;
                }

#endif

                case MlasConvAlgorithmExpandThenGemmSegmented:
                {
                    //
                    // Attempt to launch the convolution across multiple threads or fall
                    // back to a single thread.
                    //

                    if (!MlasConvTryMultithread(Parameters, Input, filter, bias, WorkingBuffer,
                        Output, ThreadPool)) {
                        MlasConvOperation(Parameters, Input, filter, bias, WorkingBuffer,
                            Output, 0, OutputSize);
                    }

                    break;
                }
            }

            //
            // Advance the buffer pointers.
            //

            if (bias != nullptr) {
                bias += FilterCount;
            }

            filter += FilterGroupSize;
            Input += InputGroupSize;
            Output += OutputGroupSize;
        }
    }
}
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
// Chance of arithmetic overflow could be reduced
#pragma warning(disable : 26451)
#endif
void
MLASCALL
MlasConvPrepare(
    MLAS_CONV_PARAMETERS* Parameters,
    size_t Dimensions,
    size_t BatchCount,
    size_t GroupCount,
    size_t InputChannels,
    const int64_t* InputShape,
    const int64_t* KernelShape,
    const int64_t* DilationShape,
    const int64_t* Padding,
    const int64_t* StrideShape,
    const int64_t* OutputShape,
    size_t FilterCount,
    const MLAS_ACTIVATION* Activation,
    size_t* WorkingBufferSize,
    float Beta,
    MLAS_THREADPOOL* ThreadPool
    )
/*++

Routine Description:

    This routine prepares for a convolution operation by computing required
    parameters including the required working buffer size for intermediate
    results.

Arguments:

    Parameters - Supplies the structure that stores the provided and computed
        parameters for the convolution operation.

    Dimensions - Supplies the number of dimensions (must be between 1 and 3).

    BatchCount - Supplies the number of batches to the processed.

    GroupCount - Supplies the number of channel groups.

    InputChannels - Supplies the number of input channels per group.

    InputShape - Supplies the shape of the input tensor.

    KernelShape - Supplies the shape of the kernel transform.

    DilationShape - Supplies the shape of the dilation.

    Padding - Supplies the number of zero padding elements at the edge of the
        input tensor.

    StrideShape - Supplies the shape of the stride.

    OutputShape - Supplies the shape of the output tensor.

    FilterCount - Supplies the number of rows of the filter matrix per group.

    Activation - Supplies the parameters for the activation to apply to the
        convolution output.

    WorkingBufferSize - Receives the number of elements to allocate for the
        working buffer for intermediate results.

    ThreadPool - Supplies the thread pool object to use, else nullptr if the
        base library threading support should be used.

Return Value:

    None.

--*/
{
    // Override
    if (GetMlasPlatform().MlasConvPrepareOverride != nullptr &&
        GetMlasPlatform().MlasConvPrepareOverride(Parameters, Dimensions, BatchCount, GroupCount, InputChannels,
        InputShape,KernelShape,DilationShape, Padding, StrideShape, OutputShape, FilterCount,
        Activation, WorkingBufferSize, Beta, ThreadPool)){
        return;
    }
    //
    // Save the convolution parameters.
    //

    Parameters->Activation = Activation;
    Parameters->BatchCount = BatchCount;
    Parameters->GroupCount = GroupCount;
    Parameters->InputChannels = InputChannels;
    Parameters->FilterCount = FilterCount;
    Parameters->Beta = Beta;

    size_t InputSize = 1;
    size_t OutputSize = 1;
    size_t K = InputChannels;

    bool AllStridesAreOne = true;
    bool AllDilationsAreOne = true;
    bool AllPaddingIsZero = true;

    for (size_t dim = 0; dim < Dimensions; dim++) {

        Parameters->InputShape[dim] = size_t(InputShape[dim]);
        Parameters->OutputShape[dim] = size_t(OutputShape[dim]);
        Parameters->KernelShape[dim] = size_t(KernelShape[dim]);
        Parameters->DilationShape[dim] = size_t(DilationShape[dim]);
        Parameters->Padding[dim] = size_t(Padding[dim]);
        Parameters->Padding[dim + Dimensions] = size_t(Padding[dim + Dimensions]);
        Parameters->StrideShape[dim] = size_t(StrideShape[dim]);

        InputSize *= Parameters->InputShape[dim];
        OutputSize *= Parameters->OutputShape[dim];
        K *= Parameters->KernelShape[dim];

        AllStridesAreOne &= (Parameters->StrideShape[dim] == 1);
        AllDilationsAreOne &= (Parameters->DilationShape[dim] == 1);
        AllPaddingIsZero &= (Parameters->Padding[dim] == 0 && Parameters->Padding[dim + Dimensions] == 0);
    }

    Parameters->InputSize = InputSize;
    Parameters->OutputSize = OutputSize;
    Parameters->K = K;

    //
    // Promote 1D convolutions to 2D convolutions.
    //

    if (Dimensions == 1) {

        Parameters->InputShape[1] = Parameters->InputShape[0];
        Parameters->InputShape[0] = 1;
        Parameters->OutputShape[1] = Parameters->OutputShape[0];
        Parameters->OutputShape[0] = 1;
        Parameters->KernelShape[1] = Parameters->KernelShape[0];
        Parameters->KernelShape[0] = 1;
        Parameters->DilationShape[1] = Parameters->DilationShape[0];
        Parameters->DilationShape[0] = 1;
        Parameters->Padding[3] = Parameters->Padding[1];
        Parameters->Padding[2] = 0;
        Parameters->Padding[1] = Parameters->Padding[0];
        Parameters->Padding[0] = 0;
        Parameters->StrideShape[1] = Parameters->StrideShape[0];
        Parameters->StrideShape[0] = 1;

        Dimensions = 2;
    }

    Parameters->Dimensions = Dimensions;

    //
    // Evaluate how the convolution will be performed.
    //

    *WorkingBufferSize = 0;

    if (AllStridesAreOne && AllPaddingIsZero) {

        //
        // Detect a pointwise convolution.
        //

        if (K == InputChannels) {

            Parameters->Algorithm = MlasConvAlgorithmGemmDirect;
            Parameters->u.GemmDirect.TransB = CblasNoTrans;
            Parameters->u.GemmDirect.ldb = OutputSize;

            return;
        }

        if (Dimensions == 2 && AllDilationsAreOne && InputChannels == 1) {

            //
            // Detect convolutions where the kernel is using the entire input
            // width or height.
            //

            if (Parameters->KernelShape[1] == Parameters->InputShape[1]) {

                Parameters->Algorithm = MlasConvAlgorithmGemmDirect;
                Parameters->u.GemmDirect.TransB = CblasTrans;
                Parameters->u.GemmDirect.ldb = Parameters->InputShape[1];

                return;
            }

            if (Parameters->KernelShape[0] == Parameters->InputShape[0] &&
                Parameters->KernelShape[1] == 1) {

                Parameters->Algorithm = MlasConvAlgorithmGemmDirect;
                Parameters->u.GemmDirect.TransB = CblasNoTrans;
                Parameters->u.GemmDirect.ldb = Parameters->InputShape[1];

                return;
            }
        }
    }

    if (MlasConvUsePyTorchIm2ColPath(Parameters)) {

        Parameters->Algorithm = MlasConvAlgorithmExpandThenGemmPyTorchIm2Col;
        *WorkingBufferSize = (OutputSize * K) + (OutputSize * FilterCount);

    } else if (FilterCount > OutputSize) {

        //
        // The filter count is larger than the output dimensions, so perform the
        // full matrix expansion and then invoke the threaded GEMM.
        //

        Parameters->Algorithm = MlasConvAlgorithmExpandThenGemm;

        *WorkingBufferSize = OutputSize * K;

    } else {

#if defined(MLAS_TARGET_WASM_SCALAR) || defined(MLAS_TARGET_ARM64) || defined(MLAS_TARGET_AMD64)

    if (Dimensions == 2
        && GroupCount > 1
        && Parameters->FilterCount == 2 && Parameters->InputChannels == 1
        && Parameters->KernelShape[0] == 7 && Parameters->KernelShape[1] == 7
        && Parameters->Padding[0] == 3 && Parameters->Padding[1] == 3
        && Parameters->Padding[2] == 3 && Parameters->Padding[3] == 3
        && Parameters->StrideShape[0] == 2 && Parameters->StrideShape[1] == 2
        && Parameters->DilationShape[0] == 1 && Parameters->DilationShape[1] == 1) {

        Parameters->Algorithm = MlasConvAlgorithmDepthwiseMultiplier2;
        return;
    }

#if defined(MLAS_TARGET_WASM_SCALAR) || defined(MLAS_TARGET_ARM64)

        // Scalar (WASM_SCALAR) / vectorized (ARM64) direct conv for depthwise convolution.
        // Currently only support 3x3 kernel with padding <=1 and dilations = 1
        // and on ARM64, it is further restricted to strides = 1.
        // TODO: support more general depthwise convolution.

        // On ARM64, only support stride = 1 for depthwise conv.
    #if defined(MLAS_TARGET_ARM64)
        bool depthwise_conv_stride_support_check = Parameters->StrideShape[0] == 1 && Parameters->StrideShape[1] == 1;
    #else
        bool depthwise_conv_stride_support_check = true;
    #endif

        if (Dimensions == 2
                && Parameters->FilterCount == 1 && Parameters->InputChannels == 1
                && Parameters->KernelShape[0] == 3 && Parameters->KernelShape[1] == 3
                && Parameters->Padding[0] <= 1 && Parameters->Padding[1] <= 1
                && Parameters->Padding[2] <= 1 && Parameters->Padding[3] <= 1
                && depthwise_conv_stride_support_check
                && Parameters->DilationShape[0] == 1 && Parameters->DilationShape[1] == 1) {

            *WorkingBufferSize = Parameters->InputShape[1] + 2;
            Parameters->Algorithm = MlasConvAlgorithmDepthwise;
            return;
        }

#endif

#endif

        //
        // Segment the operation across multiple threads by slicing the N
        // dimension (see MlasSgemmTryMultithread).
        //
        // Compute the number of target threads given the complexity of the
        // convolution operation. Small requests should run using the single
        // threaded path.
        //

        ptrdiff_t TargetThreadCount;
        double Complexity = double(FilterCount) * double(OutputSize) * double(K);

        if (Complexity < double(MLAS_SGEMM_THREAD_COMPLEXITY * MLAS_MAXIMUM_THREAD_COUNT)) {
            TargetThreadCount = ptrdiff_t(Complexity / double(MLAS_SGEMM_THREAD_COMPLEXITY)) + 1;
        } else {
            TargetThreadCount = MLAS_MAXIMUM_THREAD_COUNT;
        }

        ptrdiff_t MaximumThreadCount = MlasGetMaximumThreadCount(ThreadPool);

        if (TargetThreadCount >= MaximumThreadCount) {
            TargetThreadCount = MaximumThreadCount;
        }

        //
        // Compute the thread stride for slicing the N dimension.
        //

        size_t StrideN = OutputSize / TargetThreadCount;

        if ((StrideN * TargetThreadCount) != OutputSize) {
            StrideN++;
        }

        if (TargetThreadCount > 1) {

            StrideN = (StrideN + MLAS_SGEMM_STRIDEN_THREAD_ALIGN - 1) & ~(MLAS_SGEMM_STRIDEN_THREAD_ALIGN - 1);

            if (StrideN >= OutputSize) {
                TargetThreadCount = 1;
            } else if (StrideN * (TargetThreadCount - 1) >= OutputSize) {
                TargetThreadCount--;
            }
        }

        if (MlasConvUsePackedSgemmBPath(Parameters) && MlasConvUsePackedSgemmBTileTuning32x32(Parameters)) {
            StrideN = 32;
            TargetThreadCount = ptrdiff_t((OutputSize + StrideN - 1) / StrideN);

            if (TargetThreadCount > MaximumThreadCount) {
                TargetThreadCount = MaximumThreadCount;
            }
        }

        Parameters->ThreadCount = TargetThreadCount;

        Parameters->Algorithm = MlasConvAlgorithmExpandThenGemmSegmented;
        Parameters->u.ExpandThenGemmSegmented.ThreadStrideN = StrideN;

        *WorkingBufferSize = TargetThreadCount * MLAS_CONV_WORKING_BUFFER_SIZE_PER_THREAD;

        if (Parameters->BatchCount > 1 || Parameters->GroupCount > 1) {

            size_t WorkingBufferSizePerThread = std::max({Parameters->OutputSize * Parameters->K,
                                                          Parameters->FilterCount * Parameters->OutputSize,
                                                          static_cast<size_t>(MLAS_CONV_WORKING_BUFFER_SIZE_PER_THREAD)});
            TargetThreadCount = MaximumThreadCount;
            if (static_cast<size_t>(TargetThreadCount) >= Parameters->BatchCount * Parameters->GroupCount) {
                TargetThreadCount = static_cast<ptrdiff_t>(Parameters->BatchCount * Parameters->GroupCount);
            }
            *WorkingBufferSize = TargetThreadCount * WorkingBufferSizePerThread;
        }
    }
}
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif
