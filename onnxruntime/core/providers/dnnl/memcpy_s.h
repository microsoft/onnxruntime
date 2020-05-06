// Copyright(C) 2018 Intel Corporation
// Licensed under the MIT License

#pragma once

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

// memcpy is deprecated. Replacing it with more secure equivalent memcpy_s
//
#define MEMCPY_S(dest, src, destsz, srcsz) memcpy(dest, src, MIN(destsz, srcsz))

