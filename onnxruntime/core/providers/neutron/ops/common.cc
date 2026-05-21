// Copyright 2024-2026 NXP
// SPDX-License-Identifier: MIT

#include "core/providers/neutron/ops/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace neutron {

void PrepareForQDQ(const TensorShape& input_shape,
                   const Tensor& scale,
                   const Tensor* zero_point_ptr,
                   int64_t axis,
                   int64_t& block_count,
                   int64_t& broadcast_dim,
                   int64_t& block_size) {
  if (IsScalarOr1ElementVector(&scale)) {  // per-tensor QuantizeLinear/DequantizeLinear
    block_count = 1;
    broadcast_dim = 1;
    block_size = static_cast<size_t>(input_shape.Size());

    // enforce that zero point are scalars
    ORT_ENFORCE(zero_point_ptr == nullptr || IsScalarOr1ElementVector(zero_point_ptr),
                "x_zero_point must be null or a scalar or 1D tensor or size 1.");
  } else {  // per-channel QuantizeLinear/DequantizeLinear
    const int64_t axis_no_neg = HandleNegativeAxis(axis, input_shape.NumDimensions());
    block_count = input_shape.SizeToDimension(onnxruntime::narrow<size_t>(axis_no_neg));
    broadcast_dim = input_shape[onnxruntime::narrow<size_t>(axis_no_neg)];
    block_size = input_shape.SizeFromDimension(SafeInt<size_t>(axis_no_neg) + 1);

    // if an axis was specified, ensure the scale and zero point are compatible
    ORT_ENFORCE(scale.Shape().NumDimensions() == 1 && scale.Shape()[0] == broadcast_dim,
                "scale must be 1D tensor with size ",
                broadcast_dim);
    ORT_ENFORCE(zero_point_ptr == nullptr ||
                    (zero_point_ptr->Shape().NumDimensions() == 1 && zero_point_ptr->Shape()[0] == broadcast_dim),
                "x_zero_point must be null or 1D tensor with size ",
                broadcast_dim);
  }
}

uint32_t ScaleToNeutron(float scale_data) {
  float* scale_ptr = &scale_data;
  uint32_t u32 = *(uint32_t*)(scale_ptr);

  // Extract IEEE 754 single precision components:
  // - Bits 23-30: exponent (8 bits)
  // - Bits 9-22: mantissa (14 bits, excluding hidden bit)
  uint32_t scaler = (u32 >> 8) & 0x7fff;  // mantissa without hidden bit
  int8_t exp_tmp = (u32 >> 23) & 0xff;    // exponent with bias

  // Add hidden bit or zero out (if zero or subnormal)
  scaler = (exp_tmp == 0) ? 0 : scaler | 0x8000;
  // We subtract FP32 offset as well as 16bit growth of our scaler
  // (126 is power of -1 so mantissa is in range 0.5 to 1, 126 + 16=142,
  // where 16 is the factor we multiply by in scaler)
  exp_tmp = -(exp_tmp - 142);
  // Ensure that we don't exceed available shift bits
  // (note that this step could, in theory be skipped if this never happens.
  // Not sure if we can take the chance)
  int8_t exp = (exp_tmp > 63) ? 63 : exp_tmp;
  // Merge scaler and downshift factor into the Neutron 32bit scaler format
  // (16bit scaler in LSB and then 6bits of downshift)
  scaler = (exp << 16) | scaler;

  return scaler;
}

/// Solve tiling parameters for Neutron NPU memory layout.
///
/// Calculates optimal tiling configuration (channel density, number of neutrons,
/// divisions) to fit weight matrices and activations within Neutron TCM memory.
///
/// @param embeddings_in Input embedding dimension (K)
/// @param groupSize Quantization group size
/// @param resNumBytes Result bytes per element (typically 4 for int32)
/// @param weightBits Weight bits (4 or 8)
/// @param decodeWeights Whether weights need decoding
/// @param useDecodeBias Whether to use decode bias
/// @param MACS MAC unit size (16)
/// @param neutrons Number of Neutron cores
/// @param tcm_size TCM size in bytes (1MB default)
/// @param tcm_banks Number of TCM banks (16 default)
/// @return tuple<channelDensity_per_neutron, numNeutrons, divisions>
/// @throws std::invalid_argument if no feasible solution found
std::tuple<int, int, int>
TilingSolver(int embeddings_in, int groupSize, int resNumBytes,
             int weightBits, bool decodeWeights, bool useDecodeBias,
             int MACS, int neutrons, int tcm_size, int tcm_banks) {
  double scale = decodeWeights ? 1.0 : weightBits / 8.0;
  int channelDensity = 2 * MACS * neutrons;
  int lineDensity = 1;
  int numNeutrons = neutrons;
  bool bPingPong = true;

  auto align_to_bank = [&](double value) {
    return std::ceil(value / (tcm_size * 1.0 / tcm_banks)) * (tcm_size * 1.0 / tcm_banks);
  };

  auto calc_offsetB = [&](int cd, int nn) {
    double base = std::ceil(cd / (double)nn * embeddings_in * scale) * nn + cd * 8;
    double aligned = align_to_bank(base) / nn;

    if (decodeWeights) {
      double decode_size = cd * embeddings_in +
                           (2 + (useDecodeBias ? 1 : 0)) * cd * embeddings_in / groupSize +
                           16 * 1024 * nn + cd * 8;
      aligned = std::max(aligned, align_to_bank(decode_size) / nn);
    }

    return aligned;
  };

  double offsetB = calc_offsetB(channelDensity, numNeutrons);
  double offsetA = align_to_bank(lineDensity * embeddings_in) / numNeutrons;
  double pingpongDist = bPingPong ? offsetB : 0;
  double offsetOut = align_to_bank(channelDensity * lineDensity * resNumBytes) / numNeutrons;

  bool solved = false;
  int divisions = 1;

  while (!solved) {
    int i = 1;
    while (
        tcm_size - offsetA * numNeutrons -
                (pingpongDist + offsetB) * numNeutrons -
                channelDensity * lineDensity * resNumBytes <
            0 &&
        i <= 1) {
      i++;
      lineDensity = std::ceil(1.0 / i);
      offsetB = calc_offsetB(channelDensity, numNeutrons);
      offsetA = align_to_bank(lineDensity * embeddings_in) / numNeutrons;
      pingpongDist = bPingPong ? offsetB : 0;
    }

    if (i <= 1) {
      solved = true;
    } else {
      if (channelDensity == MACS * numNeutrons && !bPingPong && numNeutrons != 1) {
        numNeutrons = 1;
        channelDensity = 2 * MACS * numNeutrons;
        bPingPong = true;
      } else if (channelDensity == 2 * MACS * numNeutrons && bPingPong) {
        channelDensity = MACS * numNeutrons;
      } else if (channelDensity == MACS * numNeutrons && bPingPong) {
        bPingPong = false;
      } else if (channelDensity == MACS * numNeutrons && !bPingPong && numNeutrons == 1) {
        break;
      } else {
        throw std::invalid_argument("NeutronEP:MatMulCommon no feasible solution found");
      }

      lineDensity = 1;
      offsetB = calc_offsetB(channelDensity, numNeutrons);
      offsetA = align_to_bank(lineDensity * embeddings_in) / numNeutrons;
      pingpongDist = bPingPong ? offsetB : 0;
    }
  }

  if (decodeWeights && !bPingPong) {
    divisions = 2;
    lineDensity = 1;

    offsetB = align_to_bank(
                  channelDensity * embeddings_in +
                  16.0 * 1024 * numNeutrons +
                  channelDensity * embeddings_in * (2 + (useDecodeBias ? 1 : 0)) / divisions / groupSize +
                  channelDensity * 8.0) /
              numNeutrons;

    offsetA = align_to_bank(lineDensity * embeddings_in) / numNeutrons;
    offsetOut = align_to_bank(channelDensity * lineDensity * resNumBytes) / numNeutrons;

    double modPingZone = channelDensity * embeddings_in * weightBits / 8.0 / divisions +
                         channelDensity * embeddings_in / (double)groupSize * (2 + (useDecodeBias ? 1 : 0)) / divisions;

    solved = false;
    while (!solved) {
      int i = 1;
      while (
          tcm_size - offsetB * numNeutrons - offsetA * numNeutrons -
                  offsetOut * numNeutrons - modPingZone <
              0 &&
          i <= 1) {
        i++;
        lineDensity = std::ceil(1.0 / i);

        offsetB = align_to_bank(
                      channelDensity * embeddings_in +
                      16.0 * 1024 * numNeutrons +
                      channelDensity * embeddings_in * (2 + (useDecodeBias ? 1 : 0)) / divisions / groupSize +
                      channelDensity * 8.0) /
                  numNeutrons;

        offsetA = align_to_bank(lineDensity * embeddings_in) / numNeutrons;
        offsetOut = align_to_bank(channelDensity * lineDensity * resNumBytes) / numNeutrons;

        modPingZone = channelDensity * embeddings_in * weightBits / 8.0 / divisions +
                      channelDensity * embeddings_in / (double)groupSize * (2 + (useDecodeBias ? 1 : 0)) / divisions;
      }

      if (i <= 1) {
        solved = true;
      } else {
        divisions *= 2;
        lineDensity = 1;

        offsetB = align_to_bank(
                      channelDensity * embeddings_in +
                      16.0 * 1024 * numNeutrons +
                      channelDensity * embeddings_in * (2 + (useDecodeBias ? 1 : 0)) / divisions / groupSize +
                      channelDensity * 8.0) /
                  numNeutrons;

        offsetA = align_to_bank(lineDensity * embeddings_in) / numNeutrons;
        offsetOut = align_to_bank(channelDensity * lineDensity * resNumBytes) / numNeutrons;

        modPingZone = channelDensity * embeddings_in * weightBits / 8.0 / divisions +
                      channelDensity * embeddings_in / (double)groupSize * (2 + (useDecodeBias ? 1 : 0)) / divisions;
      }
    }
  }

  return std::make_tuple(channelDensity / numNeutrons, numNeutrons, divisions);
}

void OrganizeWeightsData(const int8_t* weights, int8_t* output, int rowsB,
                         int colsB, int channelDensity, int numNeutrons,
                         int weightBits, int MACs, bool isTransposed) {
  int da = 0;  // data address (read pointer in B)
  int sa = 0;  // store address (write pointer in weights_packed)

  int dstStride = channelDensity * colsB * weightBits / 8;
  int inner_cnt = MACs * MACs;
  int iters = dstStride / inner_cnt;
  int stride = dstStride - inner_cnt;
  int repeats = rowsB / channelDensity / numNeutrons;

  for (int repeat = 0; repeat < repeats; ++repeat) {
    for (int iter = 0; iter < iters; ++iter) {
      int da_save = da;

      for (int idx = 0; idx < numNeutrons; ++idx) {
        for (int jdx = 0; jdx < inner_cnt; ++jdx) {
          if (isTransposed) {
            int row = da / colsB;
            int col = da % colsB;
            output[sa++] = weights[col * rowsB + row];
          } else {
            output[sa++] = weights[da];
          }

          da++;
        }
        da += stride;  // skip to next stride
      }

      da = da_save + inner_cnt;  // restore to next base for next iter
    }

    da = da - inner_cnt * iters;
    da += channelDensity * numNeutrons * colsB * weightBits / 8;
  }
}

int32_t
GetMatmulTypeFlag(bool packed, bool signedData) {
  int32_t type = 0;
  if (packed && signedData) {
    type = 2;
  } else if (packed && !signedData) {
    type = 1;
  } else if (!packed && signedData) {
    type = -2;
  } else {
    type = -1;
  }
  return type;
}

}  // namespace neutron
}  // namespace onnxruntime
