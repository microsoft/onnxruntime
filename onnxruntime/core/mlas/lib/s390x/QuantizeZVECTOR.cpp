#include "mlasi.h"
#include <vecintrin.h>

template <typename OutputType>
void
MLASCALL
MlasQuantizeLinearZVECTOR(
    const float* Input,
    OutputType* Output,
    size_t N,
    float Scale,
    OutputType ZeroPoint
    )
{
    // Workaround for bad GCC warning that Scale is set but not used.
    MLAS_UNREFERENCED_PARAMETER(Scale);

    constexpr int32_t MinimumValue = std::numeric_limits<OutputType>::min();
    constexpr int32_t MaximumValue = std::numeric_limits<OutputType>::max();

    auto ScaleVector = vec_splats(Scale);
    auto MinimumValueVector = vec_splats(float(MinimumValue));
    auto MaximumValueVector = vec_splats(float(MaximumValue));
    auto ZeroPointVector = vec_splats(float(ZeroPoint));

    while (N >= 16) {
        auto FloatVector0 = vec_xl(0, Input);
        auto FloatVector1 = vec_xl(0, Input + 4);
        auto FloatVector2 = vec_xl(0, Input + 8);
        auto FloatVector3 = vec_xl(0, Input + 12);

        FloatVector0 /= ScaleVector;
        FloatVector1 /= ScaleVector;
        FloatVector2 /= ScaleVector;
        FloatVector3 /= ScaleVector;

        FloatVector0 = vec_round(FloatVector0);
        FloatVector1 = vec_round(FloatVector1);
        FloatVector2 = vec_round(FloatVector2);
        FloatVector3 = vec_round(FloatVector3);

        FloatVector0 += ZeroPointVector;
        FloatVector1 += ZeroPointVector;
        FloatVector2 += ZeroPointVector;
        FloatVector3 += ZeroPointVector;

        FloatVector0 = vec_max(FloatVector0, MinimumValueVector);
        FloatVector1 = vec_max(FloatVector1, MinimumValueVector);
        FloatVector2 = vec_max(FloatVector2, MinimumValueVector);
        FloatVector3 = vec_max(FloatVector3, MinimumValueVector);

        FloatVector0 = vec_min(FloatVector0, MaximumValueVector);
        FloatVector1 = vec_min(FloatVector1, MaximumValueVector);
        FloatVector2 = vec_min(FloatVector2, MaximumValueVector);
        FloatVector3 = vec_min(FloatVector3, MaximumValueVector);

        auto IntegerVector0 = vec_signed(FloatVector0);
        auto IntegerVector1 = vec_signed(FloatVector1);
        auto IntegerVector2 = vec_signed(FloatVector2);
        auto IntegerVector3 = vec_signed(FloatVector3);

        auto ShortVector0 = vec_pack(IntegerVector0, IntegerVector1);
        auto ShortVector1 = vec_pack(IntegerVector2, IntegerVector3);
        auto CharVector = vec_pack(ShortVector0, ShortVector1);
        vec_xst(CharVector, 0, (int8_t *) Output);

        // Workaround for bad GCC warning that variable is set but not used.
        MLAS_UNREFERENCED_PARAMETER(CharVector);

        Output += 16;
        Input += 16;
        N -= 16;
    }

    while (N >= 4) {
        auto FloatVector = vec_xl(0, Input);
        FloatVector /= ScaleVector;
        FloatVector = vec_round(FloatVector);
        FloatVector += ZeroPointVector;

        FloatVector = vec_max(FloatVector, MinimumValueVector);
        FloatVector = vec_min(FloatVector, MaximumValueVector);
        auto IntegerVector = vec_signed(FloatVector);

        auto ShortVector = vec_pack(IntegerVector, vec_splats((int32_t) 0));
        auto CharVector = vec_pack(ShortVector, vec_splats((int16_t) 0));

        OutputType tmp_output[sizeof(__vector float)/sizeof(OutputType)];
        vec_xst(CharVector, 0, (int8_t *) tmp_output);
        memcpy(Output, tmp_output, N);

        // Workaround for bad GCC warning that variable is set but not used.
        MLAS_UNREFERENCED_PARAMETER(CharVector);

        Output += 4;
        Input += 4;
        N -= 4;
    }

    if (N > 0) {
        float tmp_input[sizeof(__vector float) / sizeof(float)] = {};
        memcpy(tmp_input, Input, 4*N);
        auto FloatVector = vec_xl(0, &(tmp_input[0]));

        FloatVector /= ScaleVector;
        FloatVector = vec_round(FloatVector);
        FloatVector += ZeroPointVector;

        FloatVector = vec_max(FloatVector, MinimumValueVector);
        FloatVector = vec_min(FloatVector, MaximumValueVector);
        auto IntegerVector = vec_signed(FloatVector);

        auto ShortVector = vec_pack(IntegerVector, vec_splats((int32_t) 0));
        auto CharVector = vec_pack(ShortVector, vec_splats((int16_t) 0));

        OutputType tmp_output[sizeof(__vector float)/sizeof(OutputType)];
        vec_xst(CharVector, 0, (int8_t *) tmp_output);
        memcpy(Output, tmp_output, N);

        // Workaround for bad GCC warning that variable is set but not used.
        MLAS_UNREFERENCED_PARAMETER(CharVector);
    }
}

void
MLASCALL
MlasQuantizeLinearU8KernelZVECTOR(
    const float* Input,
    uint8_t* Output,
    size_t N,
    float Scale,
    uint8_t ZeroPoint
    )
{
    MlasQuantizeLinearZVECTOR<uint8_t>(Input, Output, N, Scale, ZeroPoint);
}

void
MLASCALL
MlasQuantizeLinearS8KernelZVECTOR(
    const float* Input,
    int8_t* Output,
    size_t N,
    float Scale,
    int8_t ZeroPoint
    )
{
    MlasQuantizeLinearZVECTOR<int8_t>(Input, Output, N, Scale, ZeroPoint);
}
