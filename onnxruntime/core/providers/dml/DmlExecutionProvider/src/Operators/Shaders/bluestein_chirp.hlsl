// TBUFFER is the data type to read from src and write to dst.
// Arithmetic is always done in FP32.
#if !defined(TBUFFER)
#define TBUFFER float
#endif

RWStructuredBuffer<TBUFFER> chirp : register(u0);
RWStructuredBuffer<TBUFFER> b : register(u1);

cbuffer Constants
{
    uint StartIndex;
    uint ElementCount;
    uint DFTLength;
    uint IsInverse;
};

uint NextPowerOf2(uint x)
{
    x--;
    uint y = 1;
    while (y <= x)
    {
        y <<= 1;
    }
    return y;
}


// Returns the indices for the real and complex output uav
uint2 ComputeDestIndex(uint n)
{
    return uint2(n*2, n*2+1);
}

float2 CalculateChirp(uint n, uint N, bool isInverse)
{
    static const float PI = 3.14159265f;
    float direction = isInverse ? 1 : -1;
    // chirp[n] = e^(i * direction * pi * n * n / N)
    float theta = direction * PI * n * n / N;
    return float2(cos(theta), sin(theta));
}

[numthreads(64, 1, 1)]
void BluesteinZChirp(uint3 dtid : SV_DispatchThreadId)
{
    uint n = StartIndex + dtid.x;
    if (n < ElementCount)
    {
        uint N = DFTLength;
        uint M = NextPowerOf2(2 * N - 1);

        uint2 outputIndex = ComputeDestIndex(n);
        bool isInverse = IsInverse == 1;

        if (n < N)
        {
            float2 chirp_n = CalculateChirp(n, N, isInverse);
            float2 chirp_n_conj = float2(chirp_n.x, -chirp_n.y);
            chirp[outputIndex.x] = (TBUFFER)(chirp_n.x);
            chirp[outputIndex.y] = (TBUFFER)(chirp_n.y);
            b[outputIndex.x] = (TBUFFER)(chirp_n_conj.x);
            b[outputIndex.y] = (TBUFFER)(chirp_n_conj.y);
        }
        else if (n >= M - N + 1 && n < M)
        {
            float2 chirp_n = CalculateChirp(M - n, N, isInverse);
            float2 chirp_n_conj = float2(chirp_n.x, -chirp_n.y);
            b[outputIndex.x] = (TBUFFER)(chirp_n_conj.x);
            b[outputIndex.y] = (TBUFFER)(chirp_n_conj.y);
        }
        else
        {
            b[outputIndex.x] = (TBUFFER)(0);
            b[outputIndex.y] = (TBUFFER)(0);
        }
    }
}
