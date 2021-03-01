# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
# This script helps evaluation of GPT-2 model.
import os
import logging
import torch
import random
import numpy
import time
import timeit
import math
import statistics
from pathlib import Path
from gpt2_beamsearch_helper import Gpt2BeamSearchHelper, Gpt2BeamSearchInputs
from benchmark_helper import Precision

logger = logging.getLogger(__name__)


class Gpt2Metric:
    def __init__(
        self, treatment_name, baseline_name="Torch", top_k=20, tokenize_latency=0
    ):
        assert top_k > 1 and top_k <= 100
        self.baseline = baseline_name
        self.treatment = treatment_name
        self.name: str = f"{treatment_name} vs {baseline_name}"
        self.top_k = top_k
        self.top_1_error: int = 0
        self.top_k_error: int = 0
        self.total_samples: int = 0
        self.batch_top1_error: torch.FloatTensor = None  # top 1 error for current batch
        self.batch_topk_error: torch.FloatTensor = None  # top k error for current batch
        self.tokenize_latency = tokenize_latency
        self.seq_len_latency = {}

    def print(self):
        if self.baseline != self.treatment:
            print("---")
            print(f"Metrics for {self.treatment} (baseline={self.baseline}):")
            if self.total_samples > 0:
                top_1_error_rate = 100.0 * self.top_1_error / self.total_samples
                top_k_error_rate = 100.0 * self.top_k_error / self.total_samples
                print(
                    f"Total={self.total_samples} Top1Error={self.top_1_error} ({top_1_error_rate:.2f}%) Top{self.top_k}Error={self.top_k_error} ({top_k_error_rate:.2f}%)"
                )
        else:
            print(f"Metrics for {self.treatment} (baseline):")

        if len(self.seq_len_latency) > 0:
            print("Past sequence length range and average latency:")
            total = 0
            count = 0
            averages = []
            for key in sorted(self.seq_len_latency.keys()):
                average = statistics.mean(self.seq_len_latency[key]) * 1000.0
                if key == 0:
                    org_avg = average
                    print("\t{}:         \t{:.2f} ms".format(key, average))
                else:
                    averages.append(abs(average - org_avg))
                    org_avg = average
                    print(
                        "\t[{}, {}]:\t{:.2f} ms".format(
                            2 ** key, 2 ** (key + 1) - 1, average
                        )
                    )
                total += average * len(self.seq_len_latency[key])
                count += len(self.seq_len_latency[key])
            print("Average Latency: {:.2f} ms".format(total / count))
            print("Tokenize Latency: {:.2f} ms".format(self.tokenize_latency))
            print(
                "e2e Average Latency: {:.2f} ms".format(total + self.tokenize_latency)
            )
            print("Latency Variance: {:.2f}ms".format(sum(averages) / (count - 1)))

    def start_batch(self, batch_size: int):
        self.total_samples += batch_size
        self.batch_top1_error = torch.zeros((batch_size, 1), dtype=torch.bool)
        self.batch_topk_error = torch.zeros((batch_size, 1), dtype=torch.bool)

    def eval_batch(self, baseline, treatment, past_seq_len, verbose=True):
        self._eval_topk(baseline.top_1_tokens, treatment.top_1_tokens, 1, verbose)
        self._eval_topk(
            baseline.top_k_tokens, treatment.top_k_tokens, self.top_k, verbose
        )

    def _eval_topk(self, baseline_topk, treatment_topk, top_k, verbose=True):
        if not torch.all(torch.eq(baseline_topk, treatment_topk)):
            if top_k == 1:
                if verbose:
                    print(f"Generated tokens not matched for {self.name}")
                self.batch_top1_error |= torch.eq(
                    baseline_topk, treatment_topk
                ).logical_not()
            else:
                if verbose:
                    print(
                        f"Top {top_k} tokens not matched for {self.name}. This will lead to wrong beam search results"
                    )
                self.batch_topk_error |= (
                    torch.eq(baseline_topk, treatment_topk)
                    .logical_not()
                    .sum(1)
                    .unsqueeze(dim=1)
                    > 0
                )

    def end_batch(self):
        self.top_1_error += self.batch_top1_error.sum()
        self.top_k_error += self.batch_topk_error.sum()

    def add_latency(self, past_seq_len, latency):
        key = int(math.log2(past_seq_len)) + 1 if past_seq_len > 0 else 0
        if key not in self.seq_len_latency:
            self.seq_len_latency[key] = []
        self.seq_len_latency[key].append(latency)


class Gpt2BeamSearchTester:
    def __init__(
        self,
        input_ids,
        position_ids,
        attention_mask,
        beam_select_idx,
        input_log_probs,
        input_unfinished_sents,
        prev_step_results,
        prev_step_scores,
        num_attention_heads,
        hidden_size,
        num_layer,
        beam_size,
        device,
        is_fp16=False,
        top_k=20,
        top_k_required_order=False,
    ):

        self.batch_size = input_ids.shape[0]
        self.input_length = input_ids.shape[-1]
        self.n_layer = num_layer
        self.beam_size = beam_size

        self.input_ids = input_ids
        self.position_ids = position_ids
        self.attention_mask = attention_mask

        self.beam_select_idx = beam_select_idx

        self.input_log_probs = input_log_probs
        self.input_unfinished_sents = input_unfinished_sents

        self.prev_step_results = prev_step_results
        self.prev_step_scores = prev_step_scores

        # Emtpy past state for first inference
        self.past = []
        past_shape = [
            2,
            self.batch_size,
            num_attention_heads,
            0,
            hidden_size // num_attention_heads,
        ]
        for i in range(num_layer):
            empty_past = torch.empty(past_shape).type(
                torch.float16 if is_fp16 else torch.float32
            )
            self.past.append(empty_past.to(device))

        self.last_state = None
        self.top_1_tokens = None
        self.top_k_tokens = None
        self.top_k = top_k
        self.top_k_required_order = top_k_required_order

    def get_inputs(self) -> Gpt2BeamSearchInputs:
        return Gpt2BeamSearchInputs(
            self.input_ids,
            self.position_ids,
            self.attention_mask,
            self.beam_select_idx,
            self.input_log_probs,
            self.input_unfinished_sents,
            self.prev_step_results,
            self.prev_step_scores,
            self.past,
        )

    def save_test_data(self, session, output, save_test_data_dir, test_case_id):
        from onnx import numpy_helper

        path = os.path.join(save_test_data_dir, "test_data_set_" + str(test_case_id))
        if os.path.exists(path):
            print(f"Directory {path} existed. Skip saving test data")
            return

        os.makedirs(path, exist_ok=True)

        def add_tensor(input_tensors, torch_tensor, name):
            input_tensors.append(
                numpy_helper.from_array(torch_tensor.clone().cpu().numpy(), name)
            )

        input_tensors = []
        add_tensor(input_tensors, self.input_ids, "input_ids")

        add_tensor(input_tensors, self.position_ids, "position_ids")

        add_tensor(input_tensors, self.attention_mask, "attention_mask")

        for i in range(self.n_layer):
            add_tensor(input_tensors, self.past[i], "past_" + str(i))

        for i, tensor in enumerate(input_tensors):
            with open(os.path.join(path, "input_{}.pb".format(i)), "wb") as f:
                f.write(tensor.SerializeToString())

        output_names = [output.name for output in session.get_outputs()]
        for i, name in enumerate(output_names):
            tensor = numpy_helper.from_array(
                output[i]
                if isinstance(output[i], numpy.ndarray)
                else output[i].clone().cpu().numpy()
            )
            with open(os.path.join(path, "output_{}.pb".format(i)), "wb") as f:
                f.write(tensor.SerializeToString())

        print(f"Test data saved to directory {path}")

    def update(self, output, step, device):
        """
        Update the inputs for next inference.
        """
        self.last_state = (
            torch.from_numpy(output[0])
            if isinstance(output[0], numpy.ndarray)
            else output[0].clone().detach().cpu()
        )

        self.input_ids = self.last_state.view(self.batch_size * self.beam_size, -1)

        self.beam_select_idx = (
            torch.from_numpy(output[-5]).to(device)
            if isinstance(output[-5], numpy.ndarray)
            else output[-5].clone().detach().cpu()
        )
        self.input_log_probs = (
            torch.from_numpy(output[-4]).to(device)
            if isinstance(output[-4], numpy.ndarray)
            else output[-4].clone().detach().cpu()
        )
        self.input_unfinished_sents = (
            torch.from_numpy(output[-3]).to(device)
            if isinstance(output[-3], numpy.ndarray)
            else output[-3].clone().detach().cpu()
        )
        self.prev_step_results = (
            torch.from_numpy(output[-2]).to(device)
            if isinstance(output[-2], numpy.ndarray)
            else output[-2].clone().detach().cpu()
        )
        self.prev_step_scores = (
            torch.from_numpy(output[-1]).to(device)
            if isinstance(output[-1], numpy.ndarray)
            else output[-1].clone().detach().cpu()
        )
        self.top_1_tokens = self.input_ids[0]
        self.top_k_tokens = self.last_state

        self.position_ids = (
            torch.tensor([self.input_length + step - 1])
            .unsqueeze(0)
            .repeat(self.batch_size * self.beam_size, 1)
            .to(device)
        )

        if self.attention_mask.size(0) != (self.batch_size * self.beam_size):
            self.attention_mask = self.attention_mask.repeat(
                self.batch_size * self.beam_size, 1
            )
        self.attention_mask = torch.cat(
            [
                self.attention_mask,
                torch.ones([self.batch_size * self.beam_size, 1]).type_as(
                    self.attention_mask
                ),
            ],
            1,
        ).to(device)

        self.past = []

        if isinstance(output[1], tuple):  # past in torch output is tuple
            self.past = list(output[1])
        else:
            for i in range(self.n_layer):
                past_i = (
                    torch.from_numpy(output[i + 1])
                    if isinstance(output[i + 1], numpy.ndarray)
                    else output[i + 1].clone().detach()
                )
                self.past.append(past_i.to(device))

    @staticmethod
    def is_quantized_onnx_model(onnx_model_path):
        """
        Returns True if the ONNX model is quantized.
        """
        from onnx import load

        model = load(onnx_model_path)
        from onnxruntime.quantization.quantize import __producer__ as quantize_producer

        return model.producer_name == quantize_producer

    @staticmethod
    def test_generation(
        session,
        model,
        device,
        test_inputs,
        tokenize_latency=0,
        precision=Precision.FLOAT32,
        model_class="Gpt2LMHeadModel",
        top_k=20,
        top_k_no_order=True,
        max_steps=24,
        max_inputs=0,
        verbose=False,
        save_test_data=0,
        save_test_data_dir=".",
        model_root_path=None,
    ):
        """
        Test Generation using beam search to compare PyTorch and ONNX model.
        It will print top 1 and top k errors on the given test inputs.
        """
        print(
            f"start test generation: (top_k={top_k} top_k_no_order={top_k_no_order} max_steps={max_steps} test_inputs={len(test_inputs)} max_inputs={max_inputs})"
        )
        n_layer = model.config.n_layer
        n_head = model.config.n_head
        n_embd = model.config.n_embd
        beam_size = model.config.beam_size
        eos_token_id = model.config.eos_token_id
        test_data_saved = 0

        is_float16 = precision == Precision.FLOAT16
        if is_float16:
            assert "float16" in session.get_outputs()[0].type

        # We will still use fp32 torch model as baseline when onnx model if fp16
        model.eval().to(device)

        # Allocate initial buffers for IO Binding of ONNX Runtimne. The buffer size will automatically increase later.
        init_output_shapes = Gpt2BeamSearchHelper.get_output_shapes(
            batch_size=4,
            context_len=128,
            past_sequence_length=128,
            sequence_length=32,
            beam_size=1,
            step=0,
            config=model.config,
            model_class=model_class,
        )
        output_buffers = Gpt2BeamSearchHelper.get_output_buffers(
            init_output_shapes,
            device,
            is_float16=is_float16,
        )

        baseline_name = "Torch"
        treatment_name = "Quantized Onnx" if precision == Precision.INT8 else "Onnx"
        torch_metric = Gpt2Metric(baseline_name, baseline_name, top_k, tokenize_latency)
        onnx_metric = Gpt2Metric(treatment_name, baseline_name, top_k, tokenize_latency)
        onnx_io_metric = Gpt2Metric(
            treatment_name + " with IO Binding", baseline_name, top_k, tokenize_latency
        )

        for i, inputs in enumerate(test_inputs):
            if max_inputs > 0 and i == max_inputs:
                break
            if i % 10 == 0:
                print(f"{i}")
            input_ids = inputs["input_ids"]
            position_ids = inputs["position_ids"] if "position_ids" in inputs else None
            attention_mask = (
                inputs["attention_mask"] if "attention_mask" in inputs else None
            )
            beam_select_idx = (
                inputs["beam_select_idx"] if "beam_select_idx" in inputs else None
            )
            input_log_probs = (
                inputs["input_log_probs"] if "input_log_probs" in inputs else None
            )
            input_unfinished_sents = inputs["input_unfinished_sents"]
            prev_step_results = inputs["input_ids"]
            if "prev_step_scores" in inputs:
                prev_step_scores = inputs["prev_step_scores"]
            else:
                prev_step_scores = torch.zeros([input_ids.shape[0], 1])

            onnx_runner = Gpt2BeamSearchTester(
                input_ids,
                position_ids,
                attention_mask,
                beam_select_idx,
                input_log_probs,
                input_unfinished_sents,
                prev_step_results,
                prev_step_scores,
                n_head,
                n_embd,
                n_layer,
                beam_size,
                device,
                is_float16,
                top_k,
                not top_k_no_order,
            )
            onnx_io_runner = Gpt2BeamSearchTester(
                input_ids,
                position_ids,
                attention_mask,
                beam_select_idx,
                input_log_probs,
                input_unfinished_sents,
                prev_step_results,
                prev_step_scores,
                n_head,
                n_embd,
                n_layer,
                beam_size,
                device,
                is_float16,
                top_k,
                not top_k_no_order,
            )
            torch_runner = Gpt2BeamSearchTester(
                input_ids,
                position_ids,
                attention_mask,
                beam_select_idx,
                input_log_probs,
                input_unfinished_sents,
                prev_step_results,
                prev_step_scores,
                n_head,
                n_embd,
                n_layer,
                beam_size,
                device,
                False,
                top_k,
                not top_k_no_order,
            )  # Torch model baseline is fp32

            batch_size = torch_runner.batch_size
            onnx_metric.start_batch(batch_size)
            onnx_io_metric.start_batch(batch_size)
            context_len = list(onnx_runner.input_ids.size())[1]
            with torch.no_grad():
                done = torch.zeros(batch_size, dtype=torch.bool)
                for step in range(max_steps):
                    print(f"Processing step: {step}")
                    seq_len = list(onnx_runner.input_ids.size())[1]
                    past_seq_len = list(onnx_runner.past[0].size())[3]

                    start_time = timeit.default_timer()
                    pytorch_output = Gpt2BeamSearchHelper.pytorch_inference(
                        model, torch_runner.get_inputs()
                    )
                    torch_metric.add_latency(
                        past_seq_len, timeit.default_timer() - start_time
                    )
                    torch_runner.update(pytorch_output, step, device)

                    (
                        onnx_output,
                        avg_latency_ms,
                    ) = Gpt2BeamSearchHelper.onnxruntime_inference(
                        session, onnx_runner.get_inputs(), total_runs=1
                    )
                    onnx_metric.add_latency(past_seq_len, avg_latency_ms / 1000.0)
                    onnx_runner.update(onnx_output, step, device)

                    output_shapes = Gpt2BeamSearchHelper.get_output_shapes(
                        batch_size,
                        context_len,
                        past_seq_len,
                        seq_len,
                        beam_size,
                        step,
                        model.config,
                        model_class=model_class
                    )

                    Gpt2BeamSearchHelper.auto_increase_buffer_size(
                        output_buffers, output_shapes
                    )

                    (
                        onnx_io_output,
                        avg_latency_ms,
                    ) = Gpt2BeamSearchHelper.onnxruntime_inference_with_binded_io(
                        session,
                        onnx_io_runner.get_inputs(),
                        output_buffers,
                        output_shapes,
                        total_runs=1,
                        return_numpy=False,
                        include_copy_output_latency=True,
                    )
                    onnx_io_metric.add_latency(past_seq_len, avg_latency_ms / 1000.0)

                    if test_data_saved < save_test_data:
                        onnx_io_runner.save_test_data(
                            session, onnx_io_output, save_test_data_dir, test_data_saved
                        )
                        test_data_saved += 1

                    onnx_io_runner.update(onnx_io_output, step, device)

                    onnx_metric.eval_batch(
                        torch_runner, onnx_runner, past_seq_len, verbose=verbose
                    )
                    onnx_io_metric.eval_batch(
                        torch_runner, onnx_io_runner, past_seq_len, verbose=verbose
                    )

                    done = done | (not onnx_runner.input_unfinished_sents.all())
                    if torch.all(done):
                        print("break at step: ", step)
                        break

            print(f"Totally {step+1} steps run")
            onnx_metric.end_batch()
            onnx_io_metric.end_batch()

        torch_metric.print()
        onnx_metric.print()
        onnx_io_metric.print()

        print("\tONNX")
        Gpt2BeamSearchTester.pprint_results(
            onnx_runner.prev_step_results.view(batch_size * beam_size, -1),
            onnx_runner.prev_step_scores.view(batch_size * beam_size, -1),
            pad_token_id=eos_token_id,
            eos_token_id=eos_token_id,
        )
        print("\tONNX with IO binding")
        Gpt2BeamSearchTester.pprint_results(
            onnx_io_runner.prev_step_results.view(batch_size * beam_size, -1),
            onnx_io_runner.prev_step_scores.view(batch_size * beam_size, -1),
            pad_token_id=eos_token_id,
            eos_token_id=eos_token_id,
        )

    @staticmethod
    def pprint_results(
        output_ids,
        output_scores,
        pad_token_id=None,
        eos_token_id=None,
    ):
        """
        Print test generation results.
        """
        if pad_token_id is None:
            pad_token_id = 1
        if eos_token_id is None:
            eos_token_id = 1
        if torch.is_tensor(output_ids):
            output_ids = output_ids.cpu().numpy()

        for i, sample in enumerate(output_ids):
            for j, seq in enumerate(sample):
                if isinstance(seq, numpy.ndarray) or isinstance(seq, list):
                    # remove left padding
                    for k, t in enumerate(seq):
                        if t != pad_token_id:
                            seq = seq[k:]
                            break
                    # remove EOS
                    for k, t in enumerate(seq):
                        if t == eos_token_id:
                            seq = seq[: k + 1]
                            break
                    print("-" * 40)
                    result = ",".join([str(token_id) for token_id in sample])
                    print(f">> Output {j + 1}: \t{[result]}")
                else:
                    result = ",".join([str(token_id) for token_id in sample])
                    print(f">> Output {i}: \t{result}")
                    print(f">> Scores {i}: \t{output_scores[i]}")
                    break
            print("=" * 80)
