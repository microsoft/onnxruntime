import unittest

from qmoe_prompt_runner import configure_provider, first_free_gpu_index, parse_free_gpu_indices


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


if __name__ == "__main__":
    unittest.main()
