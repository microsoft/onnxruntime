//
// Copyright (c) Microsoft Corporation. All rights reserved.
//
// This shader converts an NCHW FLOAT Tensor (BGR/RGB/GRAY) into a DX texture with channel order BGRA/BGRX/RGBA/GRAY
//

#ifdef FP16
Buffer<float> input : register(t0); // SRV
#else
StructuredBuffer<float> input : register(t0); // SRV
#endif

RWTexture2D<float4> output : register(u0); // UAV

cbuffer cbCS : register(b0)
{
    uint height;
    uint width;
};

[numthreads(16, 4, 1)]
void TensorBGR8ToSurface(uint3 globalThreadId : SV_DispatchThreadId)
{
    if (globalThreadId.x < width && globalThreadId.y < height)
    {
        float4 pixel;
        uint blockSize = height * width;
        uint threadOffset = width * globalThreadId.y + globalThreadId.x;

        pixel.b = input[threadOffset] / 255.0;
        pixel.g = input[threadOffset + blockSize] / 255.0;
        pixel.r = input[threadOffset + blockSize * 2] / 255.0;
        pixel.a = 1.0f;

        output[globalThreadId.xy] = pixel;
    }
}

[numthreads(16, 4, 1)]
void TensorRGB8ToSurface(uint3 globalThreadId : SV_DispatchThreadId)
{
    if (globalThreadId.x < width && globalThreadId.y < height)
    {
        float4 pixel;
        uint blockSize = height * width;
        uint threadOffset = width * globalThreadId.y + globalThreadId.x;

        pixel.r = input[threadOffset] / 255.0;
        pixel.g = input[threadOffset + blockSize] / 255.0;
        pixel.b = input[threadOffset + blockSize * 2] / 255.0;
        pixel.a = 1.0f;

        output[globalThreadId.xy] = pixel;
    }
}

[numthreads(16, 4, 1)]
void TensorGRAY8ToSurface(uint3 globalThreadId : SV_DispatchThreadId)
{
    if (globalThreadId.x < width && globalThreadId.y < height)
    {
        float4 pixel;
        uint threadOffset = width * globalThreadId.y + globalThreadId.x;

        pixel.b = input[threadOffset] / 255.0;
        pixel.g = pixel.b;
        pixel.r = pixel.b;
        pixel.a = 1.0;
        
        output[globalThreadId.xy] = pixel;
    }
}

[numthreads(16, 4, 1)]
void TensorBGR8ToSurfaceGRAY8(uint3 globalThreadId : SV_DispatchThreadId)
{
    if (globalThreadId.x < width && globalThreadId.y < height)
    {
        float4 pixel;
        uint blockSize = height * width;
        uint threadOffset = width * globalThreadId.y + globalThreadId.x;

        pixel.b = input[threadOffset] / 255.0;
        pixel.g = input[threadOffset + blockSize] / 255.0;
        pixel.r = input[threadOffset + blockSize * 2] / 255.0;

        float grayValue = 0.2126 * pixel.r + 0.7152 * pixel.g + 0.0722 * pixel.b;
        
        output[globalThreadId.xy] = float4(grayValue, 0.0, 0.0, 0.0);
    }
}

[numthreads(16, 4, 1)]
void TensorRGB8ToSurfaceGRAY8(uint3 globalThreadId : SV_DispatchThreadId)
{
    if (globalThreadId.x < width && globalThreadId.y < height)
    {
        float4 pixel;
        uint blockSize = height * width;
        uint threadOffset = width * globalThreadId.y + globalThreadId.x;

        pixel.r = input[threadOffset] / 255.0;
        pixel.g = input[threadOffset + blockSize] / 255.0;
        pixel.b = input[threadOffset + blockSize * 2] / 255.0;

        float grayValue = 0.2126 * pixel.r + 0.7152 * pixel.g + 0.0722 * pixel.b;
        
        output[globalThreadId.xy] = float4(grayValue, 0.0, 0.0, 0.0);
    }
}

[numthreads(16, 4, 1)]
void TensorGRAY8ToSurfaceGRAY8(uint3 globalThreadId : SV_DispatchThreadId)
{
    if (globalThreadId.x < width && globalThreadId.y < height)
    {
        float4 pixel;
        uint threadOffset = width * globalThreadId.y + globalThreadId.x;

        float grayValue = input[threadOffset] / 255.0;
        
        output[globalThreadId.xy] = float4(grayValue, 0.0, 0.0, 0.0);
    }
}