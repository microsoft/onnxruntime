# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Reference implementation of RoiAlign matching the ONNX spec (op_roi_align.py).
Used to generate expected values for the MaxModePositive test in roialign_test.cc.

Usage:
  # Generate expected values for ONNX spec max mode (default ORT behavior):
  python gen_roialign_max_mode_ref.py

  # Generate expected values for bilinear interpolation max mode (PR 7354 behavior):
  ORT_ROIALIGN_MAX_USE_BILINEAR_INTERPOLATION=1 python gen_roialign_max_mode_ref.py
"""

import numpy as np


def bilinear_interpolate_precalc(height, width, y, x):
    if y < -1.0 or y > height or x < -1.0 or x > width:
        return 0, 0, 0, 0, 0, 0, 0, 0  # pos1-4, w1-4

    y = max(y, 0.0)
    x = max(x, 0.0)
    y_low = int(y)
    x_low = int(x)

    if y_low >= height - 1:
        y_high = y_low = height - 1
        y = float(y_low)
    else:
        y_high = y_low + 1

    if x_low >= width - 1:
        x_high = x_low = width - 1
        x = float(x_low)
    else:
        x_high = x_low + 1

    ly = y - y_low
    lx = x - x_low
    hy = 1.0 - ly
    hx = 1.0 - lx
    w1 = hy * hx
    w2 = hy * lx
    w3 = ly * hx
    w4 = ly * lx

    pos1 = y_low * width + x_low
    pos2 = y_low * width + x_high
    pos3 = y_high * width + x_low
    pos4 = y_high * width + x_high
    return pos1, pos2, pos3, pos4, w1, w2, w3, w4


def roi_align_onnx_spec(
    X, rois, batch_indices, spatial_scale, output_height, output_width, sampling_ratio, mode, half_pixel
):
    n_rois = rois.shape[0]
    channels = X.shape[1]
    height = X.shape[2]
    width = X.shape[3]
    y_out = np.zeros((n_rois, channels, output_height, output_width), dtype=np.float64)

    for n in range(n_rois):
        roi_batch_ind = batch_indices[n]
        offset = 0.5 if half_pixel else 0.0
        roi_start_w = rois[n, 0] * spatial_scale - offset
        roi_start_h = rois[n, 1] * spatial_scale - offset
        roi_end_w = rois[n, 2] * spatial_scale - offset
        roi_end_h = rois[n, 3] * spatial_scale - offset

        roi_width = roi_end_w - roi_start_w
        roi_height = roi_end_h - roi_start_h

        # ONNX spec: no clamping when half_pixel is False and we're following current ORT behavior
        # (PR 7354 removed clamping)

        bin_size_h = roi_height / output_height
        bin_size_w = roi_width / output_width

        roi_bin_grid_h = int(sampling_ratio) if sampling_ratio > 0 else int(np.ceil(roi_height / output_height))
        roi_bin_grid_w = int(sampling_ratio) if sampling_ratio > 0 else int(np.ceil(roi_width / output_width))
        roi_bin_grid_h = max(roi_bin_grid_h, 1)
        roi_bin_grid_w = max(roi_bin_grid_w, 1)

        for c in range(channels):
            bottom_data = X[roi_batch_ind, c].flatten()
            for ph in range(output_height):
                for pw in range(output_width):
                    if mode == "max":
                        output_val = None
                        for iy in range(roi_bin_grid_h):
                            yy = roi_start_h + ph * bin_size_h + (iy + 0.5) * bin_size_h / roi_bin_grid_h
                            for ix in range(roi_bin_grid_w):
                                xx = roi_start_w + pw * bin_size_w + (ix + 0.5) * bin_size_w / roi_bin_grid_w
                                p1, p2, p3, p4, w1, w2, w3, w4 = bilinear_interpolate_precalc(height, width, yy, xx)
                                # Calculate both ways depending on the env var
                                if os.environ.get("ORT_ROIALIGN_MAX_USE_BILINEAR_INTERPOLATION", "0") != "0":
                                    # PR 7354 behavior: bilinear interpolation before max
                                    val = (
                                        w1 * bottom_data[p1]
                                        + w2 * bottom_data[p2]
                                        + w3 * bottom_data[p3]
                                        + w4 * bottom_data[p4]
                                    )
                                else:
                                    # ONNX spec behavior: max of individually weighted values
                                    val = max(
                                        w1 * bottom_data[p1],
                                        w2 * bottom_data[p2],
                                        w3 * bottom_data[p3],
                                        w4 * bottom_data[p4],
                                    )

                                if output_val is None:
                                    output_val = val
                                else:
                                    output_val = max(output_val, val)
                        y_out[n, c, ph, pw] = output_val if output_val is not None else 0.0
    return y_out


if __name__ == "__main__":
    import os

    # MaxModePositive test inputs
    N, C, H, W = 1, 3, 5, 5
    X = np.arange(N * C * H * W, dtype=np.float64).reshape(N, C, H, W)
    rois = np.array(
        [
            [7.0, 5.0, 7.0, 5.0],
            [-15.0, -15.0, -15.0, -15.0],
            [-10.0, 21.0, -10.0, 21.0],
            [13.0, 8.0, 13.0, 8.0],
            [-14.0, 19.0, -14.0, 19.0],
        ],
        dtype=np.float64,
    )
    batch_indices = np.array([0, 0, 0, 0, 0], dtype=np.int64)

    use_bilinear = os.environ.get("ORT_ROIALIGN_MAX_USE_BILINEAR_INTERPOLATION", "0") != "0"

    Y = roi_align_onnx_spec(
        X,
        rois,
        batch_indices,
        spatial_scale=1.0 / 16.0,
        output_height=3,
        output_width=4,
        sampling_ratio=2,
        mode="max",
        half_pixel=False,
    )

    if use_bilinear:
        print("// Expected values for MaxModePositive with bilinear interpolation max mode:")
    else:
        print("// Expected values for MaxModePositive with ONNX spec max mode:")

    vals = Y.flatten()
    for i in range(0, len(vals), 12):
        row = vals[i : i + 12]
        parts = []
        for j in range(0, len(row), 4):
            parts.append("  ".join([f"{float(v):8.4f}," for v in row[j : j + 4]]))
        print("      " + "   ".join(parts))
        if (i + 12) % 36 == 0:
            print()
