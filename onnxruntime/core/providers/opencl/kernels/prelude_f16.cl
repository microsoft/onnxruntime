// This file is not meant to be manually included!
// This file will be prepended to kernel

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#define SELECT_PREDICATE short // this is for select predicate cast
#define FLOAT half
#define FLOAT4 half4
#define CONVERT_FLOAT convert_half
#define CONVERT_FLOAT4 convert_half4
#define RI_F(image, coord) read_imageh((image), (SAMPLER), (coord))
#define WI_F(image, coord, value) write_imageh((image), (coord), (value))
