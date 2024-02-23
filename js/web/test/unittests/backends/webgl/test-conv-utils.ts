// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Tensor} from '../../../../lib/onnxjs/tensor';

/* eslint-disable no-bitwise */

// eslint-disable-next-line no-underscore-dangle
function matMul2d_(
    A: Float32Array|Float64Array, B: Float32Array|Float64Array, C: Float32Array|Float64Array, alpha: number,
    beta: number, M: number, N: number, K: number) {
  let offsetA = 0, offsetB = 0, offsetC = 0;
  for (let mm = 0; mm < M; mm++) {
    for (let nn = 0; nn < N; nn++) {
      let sum = 0;
      for (let kk = 0; kk < K; kk++) {
        sum += A[offsetA] * B[offsetB];
        offsetA += 1;
        offsetB += N;
      }
      offsetA -= K;
      offsetB -= N * K;
      C[offsetC] = alpha * sum + beta * C[offsetC];
      offsetC++;
      offsetB++;
    }
    offsetB -= N;
    offsetA += K;
  }
}

function matMul2d_tA(
    A: Float32Array|Float64Array, B: Float32Array|Float64Array, C: Float32Array|Float64Array, alpha: number,
    beta: number, M: number, N: number, K: number) {
  let offsetA = 0, offsetB = 0, offsetC = 0;
  for (let mm = 0; mm < M; mm++) {
    for (let nn = 0; nn < N; nn++) {
      let sum = 0;
      for (let kk = 0; kk < K; kk++) {
        sum += A[offsetA] * B[offsetB];
        offsetA += M;
        offsetB += N;
      }
      offsetA -= M * K;
      offsetB -= N * K;
      C[offsetC] = alpha * sum + beta * C[offsetC];
      offsetC++;
      offsetB++;
    }
    offsetB -= N;
    offsetA++;
  }
}

function matMul2d_tB(
    A: Float32Array|Float64Array, B: Float32Array|Float64Array, C: Float32Array|Float64Array, alpha: number,
    beta: number, M: number, N: number, K: number) {
  let offsetA = 0, offsetB = 0, offsetC = 0;
  for (let mm = 0; mm < M; mm++) {
    for (let nn = 0; nn < N; nn++) {
      let sum = 0;
      for (let kk = 0; kk < K; kk++) {
        sum += A[offsetA] * B[offsetB];
        offsetA += 1;
        offsetB += 1;
      }
      offsetA -= K;
      offsetB -= K;
      C[offsetC] = alpha * sum + beta * C[offsetC];
      offsetC++;
      offsetB += K;
    }
    offsetB -= N * K;
    offsetA += K;
  }
}

function matMul2d_tAtB(
    A: Float32Array|Float64Array, B: Float32Array|Float64Array, C: Float32Array|Float64Array, alpha: number,
    beta: number, M: number, N: number, K: number) {
  let offsetA = 0, offsetB = 0, offsetC = 0;
  for (let mm = 0; mm < M; mm++) {
    for (let nn = 0; nn < N; nn++) {
      let sum = 0;
      for (let kk = 0; kk < K; kk++) {
        sum += A[offsetA] * B[offsetB];
        offsetA += M;
        offsetB += 1;
      }
      offsetA -= M * K;
      offsetB -= K;
      C[offsetC] = alpha * sum + beta * C[offsetC];
      offsetC++;
      offsetB += K;
    }
    offsetB -= N * K;
    offsetA++;
  }
}

/**
 * perform matrix multiply on C = alpha * A * B + beta * C
 * @param A data of tensor A, whose shape is [M,K] or [K,M] (if transA)
 * @param B data of tensor B, whose shape is [K,N] or [N,K] (if transB)
 * @param C data of tensor C, whose shape is [M,N]
 */
export function matMul2d(
    A: Float32Array|Float64Array, B: Float32Array|Float64Array, C: Float32Array|Float64Array, transA: boolean,
    transB: boolean, alpha: number, beta: number, M: number, N: number, K: number): void {
  if (transA && transB) {
    matMul2d_tAtB(A, B, C, alpha, beta, M, N, K);
  } else if (transA) {
    matMul2d_tA(A, B, C, alpha, beta, M, N, K);
  } else if (transB) {
    matMul2d_tB(A, B, C, alpha, beta, M, N, K);
  } else {
    matMul2d_(A, B, C, alpha, beta, M, N, K);
  }
}

function im2col(
    data_im: Float32Array|Float64Array, data_col: Float32Array|Float64Array, channels: number, height: number,
    width: number, kernel_h: number, kernel_w: number, dilation_h: number, dilation_w: number, pad_t: number,
    pad_l: number, pad_b: number, pad_r: number, stride_h: number, stride_w: number) {
  const output_h = ~~((height + pad_b + pad_t - (dilation_h * (kernel_h - 1) + 1)) / stride_h) + 1;
  const output_w = ~~((width + pad_l + pad_r - (dilation_w * (kernel_w - 1) + 1)) / stride_w) + 1;

  // Fast path for zero padding and no dilation
  // From Torch, THNN_(unfolded_copy)
  if (dilation_h === 1 && dilation_w === 1 && pad_l === 0 && pad_r === 0 && pad_t === 0 && pad_b === 0) {
    for (let k = 0; k < channels * kernel_h * kernel_w; k++) {
      const nip = ~~(k / (kernel_h * kernel_w));
      const rest = k % (kernel_h * kernel_w);
      const kh = ~~(rest / kernel_w);
      const kw = rest % kernel_w;
      const dst_offset = nip * (kernel_h * kernel_w * output_h * output_w) + kh * (kernel_w * output_h * output_w) +
          kw * (output_h * output_w);
      const src_offset = nip * (height * width);
      for (let y = 0; y < output_h; y++) {
        const iy = y * stride_h + kh;
        const ix = kw;
        if (stride_w === 1) {
          data_col.set(
              data_im.subarray(src_offset + iy * width + ix, src_offset + iy * width + ix + output_w),
              dst_offset + y * output_w);
        } else {
          for (let x = 0; x < output_w; x++) {
            data_col[dst_offset + (y * output_w + x)] = data_im[src_offset + (iy * width + ix + x * stride_w)];
          }
        }
      }
    }
    return;
  }

  // Baseline
  const dkernel_h = dilation_h * (kernel_h - 1) + 1;
  const dkernel_w = dilation_w * (kernel_w - 1) + 1;

  const height_col = ~~((height + pad_t + pad_b - dkernel_h) / stride_h) + 1;
  const width_col = ~~((width + pad_l + pad_r - dkernel_w) / stride_w) + 1;

  const channels_col = channels * kernel_h * kernel_w;
  for (let c = 0; c < channels_col; ++c) {
    const w_offset = c % kernel_w;
    const h_offset = ~~(c / kernel_w) % kernel_h;
    const c_im = ~~(c / (kernel_h * kernel_w));
    for (let h = 0; h < height_col; ++h) {
      for (let w = 0; w < width_col; ++w) {
        const h_pad = h * stride_h - pad_t + h_offset * dilation_h;
        const w_pad = w * stride_w - pad_l + w_offset * dilation_w;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width) {
          data_col[(c * height_col + h) * width_col + w] = data_im[(c_im * height + h_pad) * width + w_pad];
        } else {
          data_col[(c * height_col + h) * width_col + w] = 0;
        }
      }
    }
  }
}

export function conv2d(
    Y: Tensor, X: Tensor, W: Tensor, B: Tensor|undefined, dilations: readonly number[], group: number,
    pads: readonly number[], strides: readonly number[]): void {
  const input_num = X.dims[0];
  const input_channels = X.dims[1];
  const input_height = X.dims[2];
  const input_width = X.dims[3];

  const filter_num = W.dims[0];
  const filter_channels = W.dims[1];
  const filter_height = W.dims[2];
  const filter_width = W.dims[3];
  const filter_size = filter_num * filter_channels * filter_height * filter_width;
  const kernel_shape = [filter_height, filter_width];

  const output_num = Y.dims[0];
  const output_channels = Y.dims[1];
  const output_height = Y.dims[2];
  const output_width = Y.dims[3];
  const output_size = output_num * output_channels * output_height * output_width;

  const input_image_size = input_height * input_width;
  const output_image_size = output_height * output_width;
  const kernel_size = kernel_shape[0] * kernel_shape[1];
  const X_offset = input_channels / group * input_image_size;
  const Y_offset = output_size / output_num / group;
  const W_offset = filter_size / group;
  const kernel_dim = input_channels / group * kernel_size;
  const col_buffer_size = kernel_dim * output_image_size;

  const col_buffer_data = new Float32Array(col_buffer_size);

  for (let image_id = 0; image_id < input_num; ++image_id) {
    let X_image_offset = 0;
    let Y_image_offset = 0;
    for (let group_id = 0; group_id < group; ++group_id) {
      im2col(
          X.floatData.subarray(X_image_offset + group_id * X_offset), col_buffer_data, input_channels / group,
          input_height, input_width, kernel_shape[0], kernel_shape[1], dilations[0], dilations[1], pads[0], pads[1],
          pads[2], pads[3], strides[0], strides[1]);

      matMul2d(
          W.floatData.subarray(group_id * W_offset), col_buffer_data,
          Y.floatData.subarray(Y_image_offset + group_id * Y_offset), false, false, 1, 0, filter_num / group,
          output_image_size, kernel_dim);
    }

    X_image_offset += X_offset * group;
    Y_image_offset += Y_offset * group;
  }

  // Add bias if applicable
  if (B) {
    const biasData = B.floatData;
    const outputData = Y.floatData;
    const batchSize = Y.dims[0];
    const outputChannels = Y.dims[1];
    const channelSize = Y.dims[2] * Y.dims[3];
    const dataSize = outputChannels * channelSize;
    for (let batch = 0; batch < batchSize; ++batch) {
      for (let channel = 0; channel < outputChannels; ++channel) {
        const offset = batch * dataSize + channel * channelSize;
        for (let index = 0; index < channelSize; ++index) {
          outputData[offset + index] += biasData[channel];
        }
      }
    }
  }
}
