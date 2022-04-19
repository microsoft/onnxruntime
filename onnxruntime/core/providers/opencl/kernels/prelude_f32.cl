// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file will be prepended to kernel

// The following line contribute one sample point to support the opinion: "Every OpenCL compiler is a bug ridden C
// compiler". It causes: error: internal error: could not emit constant value "abstractly"
#ifndef CONFORMANCE_could_not_emit_constant_value_abstractly
__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
#else
#define SAMPLER 14 // manually constant folded (CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST)
#endif

#define SELECT_PREDICATE int  // this is for select predicate cast
#define FLOAT float
#define FLOAT4 float4
#define CONVERT_FLOAT convert_float
#define CONVERT_FLOAT4 convert_float4
#define DIV(x,y) ((x)/(y))
#define EXP exp
#define RECIP4(x) ((FLOAT4)1/(x))
#define RI_F(image, coord) read_imagef((image), (SAMPLER), (coord))
#define WI_F(image, coord, value) write_imagef((image), (coord), (value))
