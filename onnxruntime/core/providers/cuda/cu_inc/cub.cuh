// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

//
// Header to include cub.cuh on Windows without build errors 
//

#if defined(_MSC_VER)
// undefine `small`. defined in the Windows SDK and used in the CUDA code (at least in v11.6).
// paths may vary depending on the Windows SDK/CUDA version you're using
//   defined in C:\Program Files (x86)\Windows Kits\10\Include\10.0.19041.0\shared\rpcndr.h
//     #define small char
//   used in C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\include\cub\device\dispatch\dispatch_segmented_sort.cuh
//     typename SmallAgentWarpMergeSortT::TempStorage small[segments_per_small_block];
#undef small
#endif

#include <cub/cub.cuh>
