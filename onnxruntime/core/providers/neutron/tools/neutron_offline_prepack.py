# Copyright 2024-2026 NXP
# SPDX-License-Identifier: MIT
import onnx
import numpy as np
import math
import argparse
import multiprocessing
from ctypes import *
from onnx import numpy_helper, helper

def DecimalToFixedPoint(number, integer_bits=10, fraction_bits=6):
    """Converts an unsigned decimal number to a fixed-point equivalent."""
    sign = 0
    if number < 0:
        sign =1
        number *= -1
    elif number > 2**integer_bits-1:
      number = 2**integer_bits-1
    # Split integer and fractional parts
    integer_part = int(number)
    fractional_part = number - integer_part

    first_bit_obtained = (integer_part>0)
    bits_obtained = int(math.log2(integer_part)) + 1 if (integer_part>0) else 0
    bits_left = integer_bits-bits_obtained
    shift =0
    fixed_point_scale = integer_part

    if(first_bit_obtained):
      fraction_bits = bits_left
    else:
      while(fractional_part < 0.5 and shift < 2**fraction_bits-1):
        fractional_part*=2
        shift = shift +1

    # Convert fractional part to binary
    frac_bin = ""
    for _ in range(fraction_bits):
        if(shift >= 2**fraction_bits-1):
          break
        fractional_part *= 2
        bit = int(fractional_part)
        fixed_point_scale = fixed_point_scale*2 | bit
        shift+=1
        fractional_part -= bit  # Remove the integer part
        if (fractional_part == 0 ):
          break
    # Combine integer and fractional parts to get the fixed point decimal number
    scale_10bit = fixed_point_scale & 0b1111111111
    shift_6bit = shift & 0b111111
    return((scale_10bit * 2**(-shift_6bit)) * ((-1)**sign))

def DecimalToNeutron(number, integer_bits=10, fraction_bits=6):
    """Converts an unsigned decimal number to a fixed-point binary representation on Neutron."""
    number = abs(number)
    if number > 2**integer_bits-1:
      number = 2**integer_bits-1
    # Split integer and fractional parts
    integer_part = int(number)
    fractional_part = number - integer_part

    first_bit_obtained = (integer_part>0)
    bits_obtained = int(math.log2(integer_part)) + 1 if (integer_part>0) else 0
    bits_left = integer_bits-bits_obtained
    shift =0
    fixed_point_scale = integer_part

    if(first_bit_obtained):
      fraction_bits = bits_left
    else:
      while(fractional_part < 0.5 and shift < 2**fraction_bits-1):
        fractional_part*=2
        shift = shift +1

    # Convert fractional part to binary
    frac_bin = ""
    for _ in range(fraction_bits):
        if(shift >= 2**fraction_bits-1):
          break
        fractional_part *= 2
        bit = int(fractional_part)
        fixed_point_scale = fixed_point_scale*2 | bit
        shift+=1
        fractional_part -= bit  # Remove the integer part
        if (fractional_part == 0 ):
          break
    # Combine integer and fractional parts
    scale_10bit = fixed_point_scale & 0b1111111111
    shift_6bit = shift & 0b111111
    return(np.int16(((shift_6bit << 10) | scale_10bit)-65536*(shift_6bit >= 32 )))

def WeightPacker(B, rowsB, colsB, channelDensity, weightBits = 4, MACs = 16):
  length = math.ceil(channelDensity* colsB * weightBits / 8) * rowsB/channelDensity
  packedWeights = np.zeros(int(length), dtype = np.int8)
  cell_pointer = 0
  bit_pointer = 0
  for j in range(0,rowsB,channelDensity):
      cell_pointer = math.ceil(channelDensity* colsB * weightBits / 8) * (j//channelDensity)
      bit_pointer = 0
      for i in range(0,colsB,MACs):
        for m in range(0,channelDensity):
          for k in range(0,MACs):
            if(8-bit_pointer >= weightBits):
              extracted_bits = B[j+m, i+k] & (2**weightBits-1)
              packedWeights[cell_pointer] = np.int8( packedWeights[cell_pointer] | (extracted_bits << bit_pointer) )
              bit_pointer += weightBits
              cell_pointer += bit_pointer//8
              bit_pointer = bit_pointer%8
            else:
              fitting_bits = 8-bit_pointer
              remaining_bits = weightBits - fitting_bits
              extracted_bits =  B[j+m, i+k] & (2**fitting_bits-1)
              rem_extracted_bits =  (B[j+m, i+k] >> fitting_bits) & (2**remaining_bits-1)
              packedWeights[cell_pointer] = np.int8( packedWeights[cell_pointer] | (extracted_bits << bit_pointer) )
              cell_pointer += 1
              bit_pointer =0
              packedWeights[cell_pointer] = np.int8( packedWeights[cell_pointer] | (rem_extracted_bits << bit_pointer) )
              bit_pointer+= remaining_bits

  return packedWeights

def TilingSolver(embeddings_in, weightBits=8, groupSize=-1, resNumBytes = 4,
    decodeWeights=True, useDecodeBias=True, MACS=16, neutrons=4, tcm_size=1024 * 1024, tcm_banks=16):
    numTokens = 1
    scale = weightBits / 8.0 if not decodeWeights else 1

    channelDensity = 2 * MACS * neutrons
    lineDensity = numTokens
    _numNeutrons = neutrons
    bPingPong = True

    # Avoid division by -1 or 0
    safeGroupSize = groupSize if groupSize > 0 else 1
    def calc_offsetB():
        base_offset = math.ceil(
            (math.ceil(channelDensity / _numNeutrons * embeddings_in * scale) * _numNeutrons + channelDensity * 8)
            / (tcm_size / tcm_banks)
        ) * (tcm_size / tcm_banks) / _numNeutrons

        if decodeWeights:
            decode_offset = math.ceil(
                (
                    channelDensity * embeddings_in +
                    (2 + 1 * useDecodeBias) * channelDensity * embeddings_in / safeGroupSize +
                    16 * 1024 * _numNeutrons +
                    channelDensity * 8
                ) / (tcm_size / tcm_banks)
            ) * (tcm_size / tcm_banks) / _numNeutrons
            return max(base_offset, decode_offset)
        else:
            return base_offset

    offsetB = calc_offsetB()
    offsetBias = offsetB - 8 * channelDensity
    offsetPostScale = offsetB - 4 * channelDensity

    offsetA = math.ceil(
        (lineDensity * embeddings_in) / (tcm_size / tcm_banks)
    ) * (tcm_size / tcm_banks) / _numNeutrons

    pingpongDist = offsetB if bPingPong else 0

    solved = False
    while not solved:
        i = 1
        while (
            tcm_size
            - offsetA * _numNeutrons
            - (pingpongDist + offsetB) * _numNeutrons
            - channelDensity * lineDensity * resNumBytes
            < 0 and i <= numTokens
        ):
            i += 1
            lineDensity = math.ceil(numTokens / i)

            offsetB = calc_offsetB()
            offsetBias = offsetB - 8 * channelDensity
            offsetPostScale = offsetB - 4 * channelDensity

            offsetA = math.ceil(
                (lineDensity * embeddings_in) / (tcm_size / tcm_banks)
            ) * (tcm_size / tcm_banks) / _numNeutrons

            pingpongDist = offsetB if bPingPong else 0

        if i <= numTokens:
            solved = True
        else:
            # Try alternative configurations
            if channelDensity == MACS * _numNeutrons and not bPingPong and _numNeutrons != 1:
                _numNeutrons = 1
                channelDensity = 2 * MACS * _numNeutrons
                bPingPong = True
            elif channelDensity == 2 * MACS * _numNeutrons and bPingPong:
                channelDensity = MACS * _numNeutrons
            elif channelDensity == MACS * _numNeutrons and bPingPong:
                bPingPong = False
            elif channelDensity == MACS * _numNeutrons and not bPingPong and _numNeutrons == 1:
                print("No feasible solution found.")
                break
            else:
                return

            lineDensity = numTokens
            offsetB = calc_offsetB()
            offsetBias = offsetB - 8 * channelDensity
            offsetPostScale = offsetB - 4 * channelDensity

            offsetA = math.ceil(
                (lineDensity * embeddings_in) / (tcm_size / tcm_banks)
            ) * (tcm_size / tcm_banks) / _numNeutrons

            pingpongDist = offsetB if bPingPong else 0

    return int(channelDensity / _numNeutrons), _numNeutrons, bPingPong

def FactorToNeutronScaler(float_factor):
  casted_factor = cast(pointer(c_float(float_factor)), POINTER(c_int32)).contents.value
  scaler = (((casted_factor) & 0xffffffff)>>8) & 0x7fff
  exp_tmp = (((casted_factor) & 0xffffffff)>>23) & 0xff
  if (exp_tmp == 0):
    scaler = 0
  else:
    scaler = scaler | 0x8000
  exp_tmp = -(exp_tmp -142)
  if (exp_tmp > 63):
    exp = 63
  else:
    exp = exp_tmp
  scaler = (exp<<16) | scaler
  return scaler

def ComputeDecodeScales(N, blocksPerCol, scales):
  group_sacles = scales.reshape(N, blocksPerCol)
  channel_scales = np.max(np.abs(group_sacles), axis = 1) / 16
  channel_scales_expand = np.expand_dims(channel_scales, axis = 1)
  channel_scales_repeat = np.repeat(channel_scales_expand, blocksPerCol, axis = 1)
  decode_scales = group_sacles / channel_scales_repeat

  fixed_point_process = np.vectorize(DecimalToFixedPoint)
  fixedPointScales = fixed_point_process(decode_scales)

  scaler_process = np.vectorize(FactorToNeutronScaler)
  factors = scaler_process(channel_scales)

  return fixedPointScales, factors.astype(np.uint32)

def ComputeWeightAndBias(B, decodeScales, N, blocksPerCol, groupSize):
  arr = np.asarray(B, dtype=np.uint8).reshape(-1)
  high4 = (arr >> 4) & 0x0F
  low4 = arr & 0x0F

  temp = np.empty(arr.size * 2, dtype=np.int8)
  temp[0::2] = low4
  temp[1::2] = high4
  B_int8 = temp.reshape(N, -1) - 8 #zero point is 8

  bias = np.zeros([N])
  scales_repeated = np.repeat(decodeScales, groupSize, axis = 1)
  B_decode = B_int8 * scales_repeated
  B_decode = np.clip(np.floor(B_decode + 0.5), -128, 127)
  for i in range(N):
    bias[i] = np.sum(B_decode[i,:]) * -128

  decodeBiases = np.zeros([N, blocksPerCol], np.int8)
  mask_repeated = scales_repeated < 0
  mask = decodeScales<0
  B_int8[mask_repeated] = -B_int8[mask_repeated] -1
  decodeBiases[mask]= -decodeBiases[mask] + 1

  return B_int8, decodeBiases, bias.astype(np.int32)

def FetchUnpOrganizeWeight(B, rowsB, colsB, channelDensity, MACs, weightBits, numNeutrons):
    B = B.reshape(-1)
    weights_packed = np.zeros(len(B), dtype=np.int8)
    da = 0  # source address index
    sa = 0  # destination address index
    dstStride = int(channelDensity * colsB * weightBits / 8)
    inner_cnt = int(MACs * MACs)
    iters = dstStride // inner_cnt
    stride = dstStride - inner_cnt

    repeats = int(rowsB / channelDensity / numNeutrons)
    for repeat in range(repeats):
        for iter in range(iters):
            da_save = da  # Save current source index
            for idx in range(numNeutrons):
                for jdx in range(inner_cnt):
                    weights_packed[sa] = B[da]  # Copy byte
                    sa += 1
                    da += 1
                da += stride  # Jump to next block
            da = da_save + inner_cnt  # Restore source index for next neutron group
        da = da - inner_cnt * iters  # Rewind to start of repeat
        da = da + int(channelDensity * numNeutrons * colsB * weightBits / 8)

    return weights_packed

def FetchUnpOrganizeDecodeData(decodeParam, rowsB, colsB, channelDensity,
        MACs, weightBits, numNeutrons, groupSize, isBias, divisions):
    if isBias:
        reorganized = np.zeros(decodeParam.size, dtype=np.int8)
    else:
        reorganized = np.zeros(decodeParam.size, dtype=np.int16)

    counter = 0
    for i in range(0, rowsB, channelDensity * numNeutrons):
        for division in range(divisions):
            for neutron in range(numNeutrons):
                for j in range(colsB // groupSize // divisions):
                    for row in range(channelDensity):
                        value = decodeParam[
                            i + row + neutron * channelDensity,
                            j + division * (colsB // groupSize // divisions)
                        ]
                        if isBias:
                            reorganized[counter] = value
                        else:
                            reorganized[counter] = DecimalToNeutron(value)
                        counter += 1

    dtype = np.int8 if isBias else np.int16
    scales_packed = np.zeros(len(reorganized), dtype=dtype)

    da = 0  # source index
    sa = 0  # destination index

    dstStride = int(channelDensity * colsB / groupSize / divisions)
    if isBias:
        inner_cnt = min(dstStride, 8 * 1024)
    else:
        inner_cnt = min(dstStride, 8 * 1024 // 2)

    iters = dstStride // inner_cnt
    stride = dstStride - inner_cnt
    repeats = rowsB // (channelDensity * numNeutrons)

    for repeat in range(repeats):
        for division in range(divisions):
            for iter in range(iters):
                da_save = da
                for idx in range(numNeutrons):
                    for jdx in range(inner_cnt):
                        scales_packed[sa] = reorganized[da]
                        sa += 1
                        da += 1
                    da += stride
                da = da_save + inner_cnt
            da -= inner_cnt * iters
            da += int(channelDensity * numNeutrons * colsB / groupSize / divisions)

    return scales_packed


def ConvertWeightToNeutron(cvt_args):
  b_name, B, scales, K, N, blockSize, weightBits = cvt_args
  print("Packing weight: ", b_name)
  blocksPerCol = (K + blockSize - 1) // blockSize
  channelDensity, numNeutrons, bPingPong = TilingSolver(K, weightBits = weightBits, groupSize = blockSize)

  decodeScales, factors = ComputeDecodeScales(N, blocksPerCol, scales)
  B_int8, decodeBiases, bias = ComputeWeightAndBias(B, decodeScales, N, blocksPerCol, blockSize)

  MACs = 16
  packedWeight = bytes()
  divisions = 2 if (not bPingPong) else 1
  for rows in range(0, N, channelDensity * numNeutrons):
    for cols in range(0, K, K // divisions):
      packed = WeightPacker(B_int8[rows : rows + channelDensity * numNeutrons, cols : cols + K // divisions],
              channelDensity * numNeutrons, K // divisions, channelDensity, weightBits)
      packed = FetchUnpOrganizeWeight(packed, channelDensity * numNeutrons, K // divisions, channelDensity,
              MACs, weightBits, numNeutrons)
      packedWeight = packedWeight + packed.tobytes()

  packedDecodeScales = FetchUnpOrganizeDecodeData(decodeScales, N, K, channelDensity, MACs, weightBits,
          numNeutrons, blockSize, False, divisions)
  packedDecodeBiases = FetchUnpOrganizeDecodeData(decodeBiases, N, K, channelDensity, MACs, weightBits,
          numNeutrons, blockSize, True, divisions)

  raw = packedWeight + packedDecodeBiases.tobytes() + \
        packedDecodeScales.tobytes() + bias.tobytes() + factors.tobytes()
  return (b_name, np.frombuffer(raw, dtype=np.uint8))

class Index:
    IN_A = 0
    IN_B = 1
    SCALES = 2
    ZERO_POINTS = 3
    G_IDX = 4
    BIAS = 5

def Main(args):
    model = onnx.load(args.input)
    nodes = [x for x in model.graph.node if x.op_type == "MatMulNBits"]
    initializers = {init.name: init for init in model.graph.initializer}

    cvt_args = []
    for node in nodes:
        attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
        K, N = attrs["K"], attrs["N"]
        block_size, bits = attrs["block_size"], attrs["bits"]
        if K % 16 or N % 128 or bits != 4:
            continue
        if len(node.input) > 3:
            continue

        b_name = node.input[Index.IN_B]
        b_tensor = initializers[b_name]
        b_data = numpy_helper.to_array(b_tensor)

        scales_name = node.input[Index.SCALES]
        scales_tensor = initializers[scales_name]
        scales_data = numpy_helper.to_array(scales_tensor)

        cvt_args.append((b_name, b_data, scales_data, K, N, block_size, bits))

    with multiprocessing.Pool(processes=args.jobs) as pool:
        results = pool.map(ConvertWeightToNeutron, cvt_args)

    for b_name, packaged_data in results:
        b_tensor = initializers[b_name]
        b_tensor.CopyFrom(numpy_helper.from_array(packaged_data, name=b_name))

    onnx.save(model, args.output)

def CheckArgs(value):
    if not value.endswith(".onnx"):
        raise argparse.ArgumentTypeError("Input file must end with '.onnx'.")
    return value

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline pack weights to Neutron format.")
    parser.add_argument("-i", "--input", required=True, type=CheckArgs, help="Input model file name")
    parser.add_argument("-o", "--output", type=CheckArgs, help="Output model file name")
    parser.add_argument("-j", "--jobs", type=int, default=multiprocessing.cpu_count(), help="Number of jobs")
    args = parser.parse_args()

    if (args.output == None):
        args.output = args.input.replace(".onnx", "_neutron.onnx")
    Main(args)
