#include "resize_transfrom_coordinates.h"

inline int clamp_to_edge(int v, int vmax) {
  return min(max(0, v), vmax);
}

__kernel void ResizeBilinear2D(
    __private const int2 global_size,
    __read_only image2d_t data,
    __write_only image2d_t output,
    __private const int2 input_wh,      // spatial dimential, W and H in NCHW
    __private const int2 output_wh,     // spatial dimential, W and H in NCHW
    __private const float inv_scale_x,  // 1.0/scale_x
    __private const float inv_scale_y,  // 1.0/scale_y
    __private const int trans_coord_mode) {
  const int output_cw_idx = get_global_id(0);
  const int output_bh_idx = get_global_id(1);
  if (output_cw_idx >= global_size.x || output_bh_idx >= global_size.y) return;
  const int c_idx = output_cw_idx / output_wh.x;
  const int output_w_idx = output_cw_idx % output_wh.x;
  const int b_idx = output_bh_idx / output_wh.y;
  const int output_h_idx = output_bh_idx % output_wh.y;

  // compute with feature map coordinate for weights
  FLOAT orig_coord_x;
  FLOAT orig_coord_y;
  TRANS_COORDS(trans_coord_mode, orig_coord_x, orig_coord_y, (FLOAT)output_w_idx, (FLOAT)output_h_idx, (FLOAT)inv_scale_x, (FLOAT)inv_scale_y, (FLOAT)output_wh.x, (FLOAT)output_wh.y, (FLOAT)input_wh.x, (FLOAT)input_wh.y);

  int x1 = (int)orig_coord_x;
  int y1 = (int)orig_coord_y;
  int x2 = (int)(orig_coord_x + 1);
  int y2 = (int)(orig_coord_y + 1);

  FLOAT weight_x2 = orig_coord_x - x1;
  FLOAT weight_x1 = 1 - weight_x2;
  FLOAT weight_y2 = orig_coord_y - y1;
  FLOAT weight_y1 = 1 - weight_y2;

  x1 = clamp_to_edge(x1, input_wh.x - 1);
  y1 = clamp_to_edge(y1, input_wh.y - 1);
  x2 = clamp_to_edge(x2, input_wh.x - 1);
  y2 = clamp_to_edge(y2, input_wh.y - 1);
  // convert feature map coordinate to image2d coordinate

  int base_x = mul24(c_idx, input_wh.x);
  int base_y = mul24(b_idx, input_wh.y);

  FLOAT4 p11 = RI_F(data, (int2)(base_x + x1, base_y + y1));
  FLOAT4 p12 = RI_F(data, (int2)(base_x + x1, base_y + y2));
  FLOAT4 p21 = RI_F(data, (int2)(base_x + x2, base_y + y1));
  FLOAT4 p22 = RI_F(data, (int2)(base_x + x2, base_y + y2));

  FLOAT4 v = (weight_x1 * weight_y1) * p11 + (weight_x1 * weight_y2) * p12 + (weight_x2 * weight_y1) * p21 + (weight_x2 * weight_y2) * p22;
  WI_F(output, (int2)(output_cw_idx, output_bh_idx), v);
}

__kernel void ResizeNearest2D(
    __private const int2 global_size,
    __read_only image2d_t data,
    __write_only image2d_t output,
    __private const int2 input_wh,      // spatial dimential, W and H in NCHW
    __private const int2 output_wh,     // spatial dimential, W and H in NCHW
    __private const float inv_scale_x,  // 1.0/scale_x
    __private const float inv_scale_y,  // 1.0/scale_y
    __private const int trans_coord_mode) {
  const int output_cw_idx = get_global_id(0);
  const int output_bh_idx = get_global_id(1);
  if (output_cw_idx >= global_size.x || output_bh_idx >= global_size.y) return;
  const int c_idx = output_cw_idx / output_wh.x;
  const int output_w_idx = output_cw_idx % output_wh.x;
  const int b_idx = output_bh_idx / output_wh.y;
  const int output_h_idx = output_bh_idx % output_wh.y;

  // compute with feature map coordinate for weights
  FLOAT orig_coord_x;
  FLOAT orig_coord_y;
  TRANS_COORDS(trans_coord_mode, orig_coord_x, orig_coord_y, (FLOAT)output_w_idx, (FLOAT)output_h_idx, (FLOAT)inv_scale_x, (FLOAT)inv_scale_y, (FLOAT)output_wh.x, (FLOAT)output_wh.y, (FLOAT)input_wh.x, (FLOAT)input_wh.y);

  int x1 = (int)orig_coord_x;
  int y1 = (int)orig_coord_y;

  x1 = clamp_to_edge(x1, input_wh.x - 1);
  y1 = clamp_to_edge(y1, input_wh.y - 1);
  // convert feature map coordinate to image2d coordinate

  int base_x = mul24(c_idx, input_wh.x);
  int base_y = mul24(b_idx, input_wh.y);

  FLOAT4 p11 = RI_F(data, (int2)(base_x + x1, base_y + y1));
  WI_F(output, (int2)(output_cw_idx, output_bh_idx), p11);
}
