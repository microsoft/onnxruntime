// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

// Below is a macro to compute the offset for scale and bias data for a row of X.
#ifndef LAYER_NORM_SCALE_BIAS_OFFSET
#define LAYER_NORM_SCALE_BIAS_OFFSET(broadcast_param, x_row, norm_size) \
  ((broadcast_param == 0) ? 0                                           \
                          : norm_size * (broadcast_param > 0 ? x_row / broadcast_param : x_row % (-broadcast_param)))
#endif
