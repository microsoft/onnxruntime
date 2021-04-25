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
from gpt2_tester import Gpt2Tester, Gpt2Metric
from gpt2_beamsearch_helper import Gpt2BeamSearchHelper, Gpt2BeamSearchInputs
from benchmark_helper import Precision

logger = logging.getLogger(__name__)

class Gpt2TesterFactory:
    @staticmethod
    def create_tester(tester_type="default"):
        testers = {
            "default": Gpt2Tester,
            "beam_search_step": Gpt2BeamSearchTester,
            "configurable_one_step_search": Gpt2BeamSearchTester,
        }
        w = testers[tester_type]
        return w

class Gpt2BeamSearchTester(Gpt2Tester):
    def __init__(self,
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
        super().__init__(
            input_ids,
            position_ids,
            attention_mask,
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            num_layer=num_layer,
            device=device,
            is_fp16=is_fp16,
            top_k=top_k,
            top_k_required_order=top_k_required_order
        )
        self.input_length = input_ids.shape[-1]
        self.n_layer = num_layer
        self.beam_size = beam_size

        self.beam_select_idx = beam_select_idx.to(device)

        float_type = torch.float16 if is_fp16 else torch.float32
        self.input_log_probs = input_log_probs.type(float_type).to(device)
        self.input_unfinished_sents = input_unfinished_sents.to(device)

        self.prev_step_results = prev_step_results.to(device) if prev_step_results is not None else None
        self.prev_step_scores = prev_step_scores.type(float_type).to(device)

        self.last_state = None

    def get_inputs(self) -> Gpt2BeamSearchInputs:
        return Gpt2BeamSearchInputs(
            self.input_ids,
            self.past,
            self.position_ids,
            self.attention_mask,
            self.beam_select_idx,
            self.input_log_probs,
            self.input_unfinished_sents,
            self.prev_step_results,
            self.prev_step_scores,
        )

    def update(self, output, step, device):
        """
        Update the inputs for next inference.
        """
        self.last_state = (
            torch.from_numpy(output[0]).to(device)
            if isinstance(output[0], numpy.ndarray)
            else output[0].clone().detach().cpu()
        )

        self.input_ids = self.last_state.view(self.batch_size * self.beam_size, -1).to(device)

        if self.position_ids is not None:
            input_unfinished_sents_id = -3
            self.prev_step_results = (
                torch.from_numpy(output[-2]).to(device)
                if isinstance(output[-2], numpy.ndarray)
                else output[-2].clone().detach().to(device)
            )
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
        else:
            input_unfinished_sents_id = -2

        self.beam_select_idx = (
            torch.from_numpy(output[input_unfinished_sents_id - 2]).to(device)
            if isinstance(output[input_unfinished_sents_id - 2], numpy.ndarray)
            else output[input_unfinished_sents_id - 2].clone().detach().to(device)
        )
        self.input_log_probs = (
            torch.from_numpy(output[input_unfinished_sents_id - 1]).to(device)
            if isinstance(output[input_unfinished_sents_id - 1], numpy.ndarray)
            else output[input_unfinished_sents_id - 1].clone().detach().to(device)
        )
        self.input_unfinished_sents = (
            torch.from_numpy(output[input_unfinished_sents_id]).to(device)
            if isinstance(output[input_unfinished_sents_id], numpy.ndarray)
            else output[input_unfinished_sents_id].clone().detach().to(device)
        )
        self.prev_step_scores = (
            torch.from_numpy(output[-1]).to(device)
            if isinstance(output[-1], numpy.ndarray)
            else output[-1].clone().detach().to(device)
        )
        self.top_1_tokens = self.input_ids[0]
        self.top_k_tokens = self.last_state

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
    def test_generation(session,
                        model,
                        device,
                        test_inputs,
                        precision=Precision.FLOAT32,
                        model_class="GPT2LMHeadModel_BeamSearchStep",
                        top_k=20,
                        top_k_no_order=True,
                        max_steps=24,
                        max_inputs=0,
                        verbose=False,
                        save_test_data=0,
                        save_test_data_dir="."):
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
        torch_metric = Gpt2Metric(baseline_name, baseline_name, top_k)
        onnx_metric = Gpt2Metric(treatment_name, baseline_name, top_k)
        onnx_io_metric = Gpt2Metric(
            treatment_name + " with IO Binding", baseline_name, top_k
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
            if model_class == "GPT2LMHeadModel_BeamSearchStep":
                prev_step_results = inputs["input_ids"]
            else:
                prev_step_results = None

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
            context_len = list(onnx_runner.input_ids.size())[-1]
            with torch.no_grad():
                for step in range(max_steps):
                    print(f"Processing step: {step}")
                    if model_class == "GPT2LMHeadModel_BeamSearchStep":
                        num_seq = beam_size
                        seq_len = list(onnx_runner.input_ids.size())[1]
                        past_seq_len = list(onnx_runner.past[0].size())[3]
                    else:
                        num_seq = sum(onnx_io_runner.input_unfinished_sents.view(-1).long().cpu())
                        past_seq_len = list(onnx_runner.past[0].size())[3]
                        seq_len = list(onnx_runner.input_ids.size())[-1] - past_seq_len

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

                    if model_class == "GPT2LMHeadModel_BeamSearchStep":
                        num_seq = beam_size
                    else:
                        num_seq = sum(onnx_io_runner.input_unfinished_sents.view(-1).long().cpu())
            
                    output_shapes = Gpt2BeamSearchHelper.get_output_shapes(
                        batch_size,
                        context_len,
                        past_seq_len,
                        seq_len,
                        beam_size,
                        step,
                        model.config,
                        model_class=model_class,
                        num_seq=num_seq,
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

                    if (
                        (not onnx_runner.input_unfinished_sents.any()) 
                        or (not torch_runner.input_unfinished_sents.any())
                    ):
                        print("break at step: ", step)
                        break

            print(f"Totally {step+1} steps run")
            onnx_metric.end_batch()
            onnx_io_metric.end_batch()

        torch_metric.print()
        onnx_metric.print()
        onnx_io_metric.print()

        print("\tONNX")
        if model_class == "GPT2LMHeadModel_BeamSearchStep":
            results_onnx = onnx_runner.prev_step_results.view(batch_size * beam_size, -1)
            results_onnx_io = onnx_io_runner.prev_step_results.view(batch_size * beam_size, -1)
        else:
            results_onnx = onnx_runner.input_ids.view(batch_size * beam_size, -1)
            results_onnx_io = onnx_io_runner.input_ids.view(batch_size * beam_size, -1)
        Gpt2BeamSearchTester.pprint_results(
            results_onnx,
            onnx_runner.prev_step_scores.view(batch_size * beam_size, -1),
            pad_token_id=eos_token_id,
            eos_token_id=eos_token_id,
        )
        print("\tONNX with IO binding")
        Gpt2BeamSearchTester.pprint_results(
            results_onnx_io,
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
