// This file is not meant to be manually included!

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#define SELECT_PREDICATE int // this is for select predicate cast
#define FLOAT float
#define FLOAT4 float4
#define CONVERT_FLOAT convert_float
#define CONVERT_FLOAT4 convert_float4
#define RI_F(image, coord) read_imagef((image), (SAMPLER), (coord))
#define WI_F(image, coord, value) write_imagef((image), (coord), (value))
