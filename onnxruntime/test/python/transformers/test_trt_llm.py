import torch
import tensorrt_llm

def quant_dequant_torch(weights: torch.Tensor, is_4_bit_quantization: bool):
    """
    Performs symmetric per-row quantization and dequantization on a weight tensor.

    This implementation is a pure PyTorch replacement for the original function that
    relied on a custom tensorrt_llm operator. It supports both 8-bit (int8) and
    4-bit (quint4x2 style) quantization. This version is modified to match the
    behavior of the tensorrt_llm operator which performs per-row quantization.

    Args:
        weights (torch.Tensor): The input weight tensor to be quantized.
        is_4_bit_quantization (bool): If True, performs 4-bit quantization. If False,
                                    performs 8-bit quantization.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - scales (torch.float16): The quantization scales for each row.
            - processed_q_weight (torch.int8): The packed quantized weights. For
              4-bit mode, two 4-bit values are packed into a single int8. For
              8-bit mode, this is the standard int8 quantized tensor. It is
              transposed relative to the input weights' shape.
            - dequantized_weights (torch.Tensor): The weights after being dequantized,
              restored to the original dtype and device.
    """
    # Determine quantization bits and range based on the mode
    if is_4_bit_quantization:
        # 4-bit symmetric quantization path
        q_bits = 4
        q_max = 2 ** (q_bits - 1) - 1  # 7
        q_min = -(2 ** (q_bits - 1))  # -8

        # Changed from per-column (dim=0) to per-row (dim=1) to match TRT-LLM
        max_abs_val = torch.max(torch.abs(weights), dim=1, keepdim=True).values
        max_abs_val[max_abs_val == 0] = 1.0
        scales = max_abs_val / q_max

        quant_weights = torch.round(weights / scales).clamp(q_min, q_max).to(torch.int8)

        # Pack two 4-bit integers into a single int8
        q_weights_t = quant_weights.T.contiguous()
        shape = q_weights_t.shape
        q_weights_t_reshaped = q_weights_t.view(shape[0], shape[1] // 2, 2)
        lower_nibble = q_weights_t_reshaped[..., 0]
        upper_nibble = q_weights_t_reshaped[..., 1]
        processed_q_weight = (lower_nibble & 0x0F) | (upper_nibble << 4)

    else:
        # 8-bit symmetric quantization path
        q_bits = 8
        q_max = 2 ** (q_bits - 1) - 1  # 127
        q_min = -(2 ** (q_bits - 1))  # -128

        # Changed from per-column (dim=0) to per-row (dim=1) to match TRT-LLM
        max_abs_val = torch.max(torch.abs(weights), dim=1, keepdim=True).values
        max_abs_val[max_abs_val == 0] = 1.0
        scales = max_abs_val / q_max

        quant_weights = torch.round(weights / scales).clamp(q_min, q_max).to(torch.int8)

        # For 8-bit, the processed weights are just the transposed quantized weights
        processed_q_weight = quant_weights.T.contiguous()

    # Dequantize the weights to verify and return for PyTorch-side parity check
    dequantized_weights = quant_weights.to(weights.dtype) * scales.to(weights.dtype)

    # Squeeze the scales to match the shape (in_features,) from TRT-LLM
    # TODO: processed_q_weight need interleave
    return (scales.squeeze().to(torch.float16), processed_q_weight, dequantized_weights.to(device=weights.device))


def quant_dequant_trt(weights, is_4_bit_quantization: bool = True):
    # use the test version `_symmetric_...` to get the non-interleaved weights
    type = torch.quint4x2 if is_4_bit_quantization else torch.int8
    # This import is needed to use torch.ops.trtllm._symmetric_quantize_last_axis_of_batched_matrix()
    # Comment out this line for passing the lintrunner check in the CI.

    quant_weights, processed_q_weight, torch_weight_scales = (
        torch.ops.trtllm._symmetric_quantize_last_axis_of_batched_matrix(weights.T.cpu().contiguous(), type)
    )

    # Unpack the int4s int int8s
    if is_4_bit_quantization:
        upper = quant_weights >> 4
        lower = (quant_weights << 4) >> 4  # Arithmetic right shift sign extends
        quant_weights = torch.stack((lower, upper), dim=2).view(weights.T.shape)

    quant_weights = quant_weights.to(dtype=weights.dtype)
    result = torch.multiply(quant_weights, torch_weight_scales.unsqueeze(0)).T.contiguous()
    return torch_weight_scales.to(torch.float16), processed_q_weight, result.to(device=weights.device)


def run_test(in_features, out_features, is_4_bit):
    """
    Runs a single test case, comparing the outputs of the two functions.
    """
    print(f"Running test for in_features={in_features}, out_features={out_features}, is_4_bit={is_4_bit}")

    # Create a random weight tensor
    weights = torch.randn(in_features, out_features, dtype=torch.float16)

    # Get the outputs from the PyTorch implementation
    scales_torch, processed_q_weight_torch, dequantized_torch = quant_dequant_torch(weights, is_4_bit)

    # Get the outputs from the TRT implementation
    scales_trt, processed_q_weight_trt, dequantized_trt = quant_dequant_trt(weights, is_4_bit)

    print(
        f"{weights.shape=} {in_features=} {out_features=} {scales_torch.shape=} {scales_trt.shape=} {processed_q_weight_torch.shape=} {processed_q_weight_trt.shape=} {dequantized_torch.shape=} {dequantized_trt.shape=}"
    )

    print("scales_torch", scales_torch)
    print("scales_trt", scales_trt)

    print("processed_q_weight_torch", processed_q_weight_torch)
    print("processed_q_weight_trt", processed_q_weight_trt)

    print("dequantized_torch", dequantized_torch)
    print("dequantized_trt", dequantized_trt)

    # Compare the processed quantized weights
    processed_q_weight_all_close = torch.allclose(processed_q_weight_torch, processed_q_weight_trt)
    print(f"  Processed quantized weights are close: {processed_q_weight_all_close}")

    # Compare the dequantized weights
    dequantized_all_close = torch.allclose(dequantized_torch, dequantized_trt, atol=1e-2)
    print(f"  Dequantized weights are close: {dequantized_all_close}")

    # Compare the scales
    scales_all_close = torch.allclose(scales_torch, scales_trt, atol=1e-2)
    print(f"  Scales are close: {scales_all_close}")

    print("-" * 30)


if __name__ == "__main__":
    # Test with 4-bit quantization
    # run_test(64, 256, is_4_bit=True)

    # Test with 8-bit quantization
    run_test(64, 256, is_4_bit=False)
