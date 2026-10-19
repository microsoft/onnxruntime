import contextlib
import io
import unittest

import numpy as np
from qmoe_prompt_runner import (
    configure_provider,
    first_free_gpu_index,
    generate,
    generation_search_options,
    parse_free_gpu_indices,
)


class RecordingConfig:
    def __init__(self):
        self.providers = ["configured"]
        self.provider_options = {}

    def clear_providers(self):
        self.providers.clear()

    def append_provider(self, provider):
        self.providers.append(provider)

    def set_provider_option(self, provider, key, value):
        self.provider_options[(provider, key)] = value


class TestParseFreeGpuIndices(unittest.TestCase):
    def test_returns_sorted_free_gpu_indices(self):
        output = """\
2, 0
0, 1024
1, 0
"""
        self.assertEqual(parse_free_gpu_indices(output), [1, 2])

    def test_returns_empty_list_when_all_gpus_are_used(self):
        self.assertEqual(parse_free_gpu_indices("0, 1\n1, 2048\n"), [])

    def test_rejects_malformed_output(self):
        with self.assertRaisesRegex(ValueError, "Unexpected nvidia-smi output"):
            parse_free_gpu_indices("0\n")


class TestFirstFreeGpuIndex(unittest.TestCase):
    def test_selects_lowest_index(self):
        self.assertEqual(first_free_gpu_index([5, 2, 7]), 2)

    def test_raises_when_no_gpu_is_free(self):
        with self.assertRaisesRegex(RuntimeError, "No free GPU found"):
            first_free_gpu_index([])


class TestConfigureProvider(unittest.TestCase):
    def test_cuda_forces_requested_attention_backend(self):
        config = RecordingConfig()
        configure_provider(config, "cuda", 1)
        self.assertEqual(config.providers, ["cuda"])
        self.assertEqual(config.provider_options, {("cuda", "sdpa_kernel"): "1"})

    def test_cpu_has_no_provider_options(self):
        config = RecordingConfig()
        configure_provider(config, "cpu", 1)
        self.assertEqual(config.providers, [])
        self.assertEqual(config.provider_options, {})

    def test_follow_config_preserves_existing_configuration(self):
        config = RecordingConfig()
        configure_provider(config, "follow_config", 1)
        self.assertEqual(config.providers, ["configured"])
        self.assertEqual(config.provider_options, {})


class TestGenerationSearchOptions(unittest.TestCase):
    def test_limits_new_tokens_relative_to_prompt(self):
        self.assertEqual(
            generation_search_options(prompt_token_count=17, max_new_tokens=256),
            {"do_sample": False, "max_length": 273},
        )

    def test_until_eos_uses_model_context_limit(self):
        self.assertEqual(
            generation_search_options(prompt_token_count=17, max_new_tokens=None),
            {"do_sample": False},
        )


class TestGenerationMarkers(unittest.TestCase):
    def test_marks_single_token_prefill_and_decode_separately(self):
        class GenerationParams:
            def __init__(self, model):
                self.options = {}

            def set_search_options(self, **options):
                self.options = options

        class Generator:
            def __init__(self, model, params):
                self.iteration = 0

            def append_tokens(self, tokens):
                print("append_tokens", file=stderr)

            def is_done(self):
                return self.iteration == 2

            def generate_next_token(self):
                print(f"generate_next_token {self.iteration}", file=stderr)
                self.iteration += 1

            def get_next_tokens(self):
                return np.array([self.iteration])

        class Genai:
            GeneratorParams = GenerationParams

        Genai.Generator = Generator

        class Tokenizer:
            def encode(self, prompt):
                return list(range(len(prompt)))

            def decode(self, tokens):
                return str(tokens)

        for token_count in (1, 3):
            with self.subTest(token_count=token_count):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    result = generate(Genai, None, Tokenizer(), "x" * token_count, 2, raw_prompt=True)
                self.assertEqual(result["prompt_tokens"], token_count)
                self.assertEqual(
                    stderr.getvalue().splitlines(),
                    [
                        "[qmoe_prompt_runner] prefill 0",
                        "append_tokens",
                        "generate_next_token 0",
                        "[qmoe_prompt_runner] decode 1",
                        "generate_next_token 1",
                    ],
                )


if __name__ == "__main__":
    unittest.main()
