import platform
import unittest

import torch
from parameterized import parameterized
from test_flash_attn_cuda import (
    Formats,
    gqa_no_past_flash_attention_test_cases,
    gqa_past_flash_attention_test_cases,
    parity_check_gqa_past,
    parity_check_gqa_past_no_buff,
    parity_check_gqa_prompt,
    parity_check_gqa_prompt_no_buff,
)

import onnxruntime


@unittest.skipIf(
    (not torch.cuda.is_available())
    or (platform.system() != "Linux")
    or ("ROCMExecutionProvider" not in onnxruntime.get_available_providers()),
    reason="ROCm is not available, skipping tests.",
)
class TestRocmGQA(unittest.TestCase):
    @parameterized.expand(gqa_no_past_flash_attention_test_cases())
    def test_gqa_no_past_flash_attention(self, _, config, local, rotary, rotary_interleaved, packed, softcap):
        config.ep = "ROCMExecutionProvider"
        print("------- FLASH ATTENTION (PROMPT CASE) --------")

        parity_check_gqa_prompt(
            config,
            local=local,
            past_format=Formats.BNSH,
            rotary=rotary,
            rotary_interleaved=rotary_interleaved,
            packed=packed,
            rtol=0.001,
            atol=0.005,
        )
        parity_check_gqa_prompt_no_buff(
            config,
            local=local,
            past_format=Formats.BNSH,
            rotary=rotary,
            rotary_interleaved=rotary_interleaved,
            packed=packed,
            rtol=0.001,
            atol=0.005,
        )

    @parameterized.expand(gqa_past_flash_attention_test_cases())
    def test_gqa_past_flash_attention(self, _, config, local, rotary, rotary_interleaved, packed, softcap):
        config.ep = "ROCMExecutionProvider"
        print("------- FLASH ATTENTION (TOKEN GEN) -------")

        parity_check_gqa_past(
            config,
            local=local,
            past_format=Formats.BNSH,
            rotary=rotary,
            rotary_interleaved=rotary_interleaved,
            packed=packed,
            rtol=0.001,
            atol=0.005,
        )
        parity_check_gqa_past_no_buff(
            config,
            local=local,
            past_format=Formats.BNSH,
            rotary=rotary,
            rotary_interleaved=rotary_interleaved,
            packed=packed,
            rtol=0.001,
            atol=0.005,
        )


if __name__ == "__main__":
    unittest.main()
