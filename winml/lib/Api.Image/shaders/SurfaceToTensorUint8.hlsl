//
// Copyright (c) Microsoft Corporation. All rights reserved.
//
// This shader converts a DX texture (BGRA/BGRX) into NCHW UINT8 Tensor with channel order RGB
// Note that this shader requires that width be a multiple of 4 because UAV loads are limited to UINT and it writes 4 UINT8 values per channel at a time
//

Texture2D input : register(t0); // SRV
RWBuffer<uint> output : register(u0); // UAV
cbuffer cbCS : register(b0)
{
    uint g_height;
    uint g_width;
};

static uint g_blkwdt = g_width / 4; // Blocks per line
static uint g_blkchn = g_height * g_blkwdt; // Blocks per channel

[numthreads(3, 1, 1)]
void main(uint gi : SV_GroupIndex, uint3 gid : SV_GroupID)
{
    uint outid = gi * g_blkchn + gid.x;
    uint inpid_x = (gid.x % g_blkwdt) * 4;
    uint inpid_y = gid.x / g_blkwdt;
    output[outid]  = clamp(uint4(input.Load(uint3(inpid_x + 0, inpid_y, 0)) * 255), 0, 255)[gi];
    output[outid] |= clamp(uint4(input.Load(uint3(inpid_x + 1, inpid_y, 0)) * 255), 0, 255)[gi] << 8;
    output[outid] |= clamp(uint4(input.Load(uint3(inpid_x + 2, inpid_y, 0)) * 255), 0, 255)[gi] << 16;
    output[outid] |= clamp(uint4(input.Load(uint3(inpid_x + 3, inpid_y, 0)) * 255), 0, 255)[gi] << 24;
}