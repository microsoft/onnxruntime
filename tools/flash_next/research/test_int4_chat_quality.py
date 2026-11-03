import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from check_int4_chat_quality import run_case, run_suite


class QualitySmokeTest(unittest.TestCase):
    def run_fake(self, generated, eos_tokens):
        class Params:
            def __init__(self, model):
                self.options = None

            def set_search_options(self, **options):
                self.options = options
                self_test.assertNotIn("min_length", options)
                self_test.assertNotIn("repetition_penalty", options)

        class Generator:
            def __init__(self, model, params):
                self.output = []

            def append_tokens(self, tokens):
                self.prompt = list(tokens)

            def is_done(self):
                return len(self.output) == len(generated)

            def generate_next_token(self):
                self.output.append(generated[len(self.output)])

            def get_next_tokens(self):
                return [self.output[-1]]

            def get_sequence(self, batch):
                output = self.output[:-1] if self.output and self.output[-1] in eos_tokens else self.output
                return self.prompt + output

        self_test = self
        og = SimpleNamespace(GeneratorParams=Params, Generator=Generator)
        tokenizer = SimpleNamespace(encode=lambda text: iter([1, 2]), decode=lambda ids: ",".join(map(str, ids)))
        result = run_case(og, object(), tokenizer, "prompt", 16, eos_tokens)
        json.dumps(result)
        return result

    def test_stops_on_eos_without_decoding_eos(self):
        result = self.run_fake([4, 99, 5], [99])
        self.assertEqual(result["output_tokens"], [4, 99])
        self.assertEqual(result["output_text"], "4")
        self.assertTrue(result["stopped_on_eos"])

    def test_max_length_does_not_drop_final_non_eos_token(self):
        result = self.run_fake([4, 5], [99])
        self.assertEqual(result["output_tokens"], [4, 5])
        self.assertEqual(result["output_text"], "4,5")
        self.assertFalse(result["stopped_on_eos"])

    def test_suite_reuses_model_for_all_prompt_modes(self):
        model = object()
        samples = []
        with (patch("check_int4_chat_quality.render_chat", side_effect=lambda _, text: "chat:" + text),
              patch("check_int4_chat_quality.run_case",
                    side_effect=lambda *args: {"output_text": "Paris"}) as run):
            run_suite(object(), model, object(), "template", [99], samples.append)
        self.assertEqual(len(samples), 9)
        self.assertTrue(all(call.args[1] is model for call in run.call_args_list))
        self.assertEqual([call.args[4] for call in run.call_args_list], [16] * 6 + [1] * 3)
        self.assertEqual(sum(sample["chat_template_applied"] for sample in samples), 6)


if __name__ == "__main__":
    unittest.main()
