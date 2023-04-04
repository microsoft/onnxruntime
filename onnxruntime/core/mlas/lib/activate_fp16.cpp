/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    activate_fp16.cpp

Abstract:

    This module implements the activation routines for fp16 data types

--*/

#include "fp16_common.h"

#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED

//
// Templates for activation functions.
//

template<MLAS_ACTIVATION_KIND ActivationKind>
struct MLAS_HALF_ACTIVATION_FUNCTION;

template<>
struct MLAS_HALF_ACTIVATION_FUNCTION<MlasReluActivation>
{
    const MLAS_FLOAT16X8 ZeroVec = MlasZeroFloat16x8();

    MLAS_HALF_ACTIVATION_FUNCTION(const MLAS_ACTIVATION& Activation)
    {
        MLAS_UNREFERENCED_PARAMETER(Activation);
    }

    MLAS_FLOAT16X8 Activate(MLAS_FLOAT16X8 Value)
    {
        return MlasMaximumFloat16x8(ZeroVec, Value);
    }

    MLAS_FLOAT16X4 Activate(MLAS_FLOAT16X4 Value)
    {
        return MlasMaximumFloat16x4(MlasToLowHalfFloat16x4(ZeroVec), Value);
    }
};

template<>
struct MLAS_HALF_ACTIVATION_FUNCTION<MlasLeakyReluActivation>
{
    const MLAS_FLOAT16X8 ZeroVec = MlasZeroFloat16x8();

    MLAS_FLOAT16X8 AlphaBroadcast;

    MLAS_HALF_ACTIVATION_FUNCTION(const MLAS_ACTIVATION& Activation)
    {
        const _mlas_fp16_ alpha = MLAS_Float2Half(Activation.Parameters.LeakyRelu.alpha);
        AlphaBroadcast = MlasBroadcastFloat16x8(alpha);
    }

    MLAS_FLOAT16X8 Activate(MLAS_FLOAT16X8 Value)
    {
        MLAS_FLOAT16X8 ValueTimesAlpha = MlasMultiplyFloat16x8(Value, AlphaBroadcast);
        return MlasBitwiseSelectFloat16x8(MlasCmpLessEqualFloat16x8(Value, ZeroVec),
                                          ValueTimesAlpha, Value);
    }

    MLAS_FLOAT16X4 Activate(MLAS_FLOAT16X4 Value)
    {
        MLAS_FLOAT16X4 ValueTimesAlpha =
            MlasMultiplyFloat16x4(Value, MlasToLowHalfFloat16x4(AlphaBroadcast));
        return MlasBitwiseSelectFloat16x4(
            MlasCmpLessEqualFloat16x4(Value, MlasToLowHalfFloat16x4(ZeroVec)), ValueTimesAlpha,
            Value);
    }
};

template<>
struct MLAS_HALF_ACTIVATION_FUNCTION<MlasClipActivation>
{
    MLAS_FLOAT16X8 MinimumBroadcast;
    MLAS_FLOAT16X8 MaximumBroadcast;

    MLAS_HALF_ACTIVATION_FUNCTION(const MLAS_ACTIVATION& Activation)
    {
        const _mlas_fp16_ min = MLAS_Float2Half(Activation.Parameters.Clip.minimum);
        MinimumBroadcast = MlasBroadcastFloat16x8(min);
        const _mlas_fp16_ max = MLAS_Float2Half(Activation.Parameters.Clip.maximum);
        MaximumBroadcast = MlasBroadcastFloat16x8(max);
    }

    MLAS_FLOAT16X8 Activate(MLAS_FLOAT16X8 Value)
    {
        Value = MlasMaximumFloat16x8(MinimumBroadcast, Value);
        Value = MlasMinimumFloat16x8(MaximumBroadcast, Value);

        return Value;
    }

    MLAS_FLOAT16X4 Activate(MLAS_FLOAT16X4 Value)
    {
        Value = MlasMaximumFloat16x4(MlasToLowHalfFloat16x4(MinimumBroadcast), Value);
        Value = MlasMinimumFloat16x4(MlasToLowHalfFloat16x4(MaximumBroadcast), Value);
        return Value;
    }
};

template<>
struct MLAS_HALF_ACTIVATION_FUNCTION<MlasHardSigmoidActivation>
{
    MLAS_FLOAT16X8 AlphaBroadcast;
    MLAS_FLOAT16X8 BetaBroadcast;
    MLAS_FLOAT16X8 MinimumBroadcast;
    MLAS_FLOAT16X8 MaximumBroadcast;

    MLAS_HALF_ACTIVATION_FUNCTION(const MLAS_ACTIVATION& Activation)
    {
        const _mlas_fp16_ alpha = MLAS_Float2Half(Activation.Parameters.HardSigmoid.alpha);
        AlphaBroadcast = MlasBroadcastFloat16x8(alpha);
        const _mlas_fp16_ beta = MLAS_Float2Half(Activation.Parameters.HardSigmoid.beta);
        BetaBroadcast = MlasBroadcastFloat16x8(beta);
        MinimumBroadcast = MlasZeroFloat16x8();
        MaximumBroadcast = MlasBroadcastFloat16x8(MLAS_Float2Half(1.0f));
    }

    MLAS_FLOAT16X8 Activate(MLAS_FLOAT16X8 Value)
    {
        Value = MlasMultiplyAddFloat16x8(Value, AlphaBroadcast, BetaBroadcast);
        Value = MlasMinimumFloat16x8(MaximumBroadcast, Value);
        Value = MlasMaximumFloat16x8(MinimumBroadcast, Value);

        return Value;
    }

    MLAS_FLOAT16X4 Activate(MLAS_FLOAT16X4 Value)
    {
        Value = MlasMultiplyAddFloat16x4(Value, MlasToLowHalfFloat16x4(AlphaBroadcast),
                                         MlasToLowHalfFloat16x4(BetaBroadcast));
        Value = MlasMinimumFloat16x4(MlasToLowHalfFloat16x4(MaximumBroadcast), Value);
        Value = MlasMaximumFloat16x4(MlasToLowHalfFloat16x4(MinimumBroadcast), Value);

        return Value;
    }
};

template<MLAS_ACTIVATION_KIND ActivationKind>
inline
void
MlasActivationKernel(
    const MLAS_ACTIVATION& Activation,
    MLAS_FP16* Buffer,
    size_t StartM,
    size_t StartN,
    size_t CountM,
    size_t CountN,
    size_t ldc
    )
{
    MLAS_HALF_ACTIVATION_FUNCTION<ActivationKind> ActivationFunction(Activation);

    auto* CRow = reinterpret_cast<_mlas_fp16_*>(Buffer);
    CRow += StartM * ldc + StartN;

    while (CountM-- > 0) {
        _mlas_fp16_* buffer = CRow;
        size_t n = CountN;

        while (n >= 8) {
            MLAS_FLOAT16X8 Vector = MlasLoadFloat16x8(buffer);
            MlasStoreFloat16x8(buffer, ActivationFunction.Activate(Vector));
            buffer += 8;
            n -= 8;
        }

        if (n >= 4) {
            MLAS_FLOAT16X4 Vector = MlasLoadFloat16x4(buffer);
            MlasStoreFloat16x4(buffer, ActivationFunction.Activate(Vector));
            buffer += 4;
            n -= 4;
        }

        if (n > 0) {
            MLAS_FLOAT16X4 buf;
            std::memcpy(&buf, buffer, n * sizeof(_mlas_fp16_));
            MLAS_FLOAT16X4 res = ActivationFunction.Activate(buf);
            MlasStorePartialFloat16x4(buffer, res, n);
        }

        CRow += ldc;
    }
}

template<>
inline
void
MlasActivationKernel<MlasIdentityActivation>(
    const MLAS_ACTIVATION& Activation,
    MLAS_FP16* Buffer,
    size_t StartM,
    size_t StartN,
    size_t CountM,
    size_t CountN,
    size_t ldc
    )
{
    //
    // No operation.
    //

    MLAS_UNREFERENCED_PARAMETER(Activation);
    MLAS_UNREFERENCED_PARAMETER(Buffer);
    MLAS_UNREFERENCED_PARAMETER(StartM);
    MLAS_UNREFERENCED_PARAMETER(StartN);
    MLAS_UNREFERENCED_PARAMETER(CountM);
    MLAS_UNREFERENCED_PARAMETER(CountN);
    MLAS_UNREFERENCED_PARAMETER(ldc);
}


void
MLAS_HALF_GEMM_ACTIVATION_PROCESSOR::Process(
    MLAS_FP16* C,
    size_t StartM,
    size_t StartN,
    size_t CountM,
    size_t CountN,
    size_t ldc
    ) const
{
    switch (Activation_.ActivationKind) {
        case MlasIdentityActivation: {
            MlasActivationKernel<MlasIdentityActivation>(Activation_, C, StartM, StartN, CountM, CountN, ldc);
            break;
        }

        case MlasReluActivation: {
            MlasActivationKernel<MlasReluActivation>(Activation_, C, StartM, StartN, CountM, CountN,
                                                     ldc);
            break;
        }

        case MlasLeakyReluActivation: {
            MlasActivationKernel<MlasLeakyReluActivation>(Activation_, C, StartM, StartN, CountM,
                                                          CountN, ldc);
            break;
        }

        case MlasClipActivation: {
            MlasActivationKernel<MlasClipActivation>(Activation_, C, StartM, StartN, CountM, CountN,
                                                     ldc);
            ;
            break;
        }

        case MlasHardSigmoidActivation: {
            MlasActivationKernel<MlasHardSigmoidActivation>(Activation_, C, StartM, StartN, CountM,
                                                            CountN, ldc);
            break;
        }

/* case MlasTanhActivation : {
            if (N == ldc) {
                MlasComputeTanh(Buffer, Buffer, M * N);
            } else {
                while (M-- > 0) {
                    MlasComputeTanh(Buffer, Buffer, N);
                    Buffer += ldc;
                }
            }

            break;
        }

        case MlasLogisticActivation: {
            if (N == ldc) {
                MlasComputeLogistic(Buffer, Buffer, M * N);
            } else {
                while (M-- > 0) {
                    MlasComputeLogistic(Buffer, Buffer, N);
                    Buffer += ldc;
                }
            }

            break;
        }
*/
        default:
            // Tanh and Logistic activation not supported.
            return;
    }
}

#else
// Really dumb implementation when fp16 acceleration is not supported

#include <vector>

MLAS_FORCEINLINE
void
CvtFloat2Half(
    _mlas_fp16_* dest,
    const float* src,
    size_t len
)
{
    for (size_t i = 0; i < len; i++) {
        *dest++ = MLAS_Float2Half(*src++);
    }
}

void
MLAS_HALF_GEMM_ACTIVATION_PROCESSOR::Process(
    MLAS_FP16* C,
    size_t StartM,
    size_t StartN,
    size_t CountM,
    size_t CountN,
    size_t ldc
    ) const
{
    std::vector<float> buffer(CountM*CountN);
    MLAS_HALF_GEMM_2FLOAT_PROCESSOR proc(this->Activation_, buffer.data(), CountN);
    proc.Process(C, StartM, StartN, CountM, CountN, ldc);

    _mlas_fp16_* Output = reinterpret_cast<_mlas_fp16_*>(C);
    const auto* CRow = buffer.data();
    Output += StartM * ldc + StartN;

    while (CountM-- > 0) {
        CvtFloat2Half(Output, CRow, CountN);
        CRow += CountN;
        Output += ldc;
    }
}

#endif  // MLAS_F16VEC_INTRINSICS_SUPPORTED
