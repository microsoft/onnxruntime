# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
# This script helps onnx conversion and validation for GPT2 model with past state.
import os
import logging
import torch
import onnx
import random
import numpy
import time
import re
from pathlib import Path
from typing import List, Dict, Tuple, Union
from transformers import GPT2LMHeadModel, GPT2Config
from benchmark_helper import Precision
from gpt2_helper import Gpt2Helper, Gpt2Inputs, GPT2ModelNoPastState, MyGPT2Model, MyGPT2LMHeadModel, MyGPT2LMHeadModel_NoPadding

logger = logging.getLogger(__name__)

BIG_NEG = -1e4

class Gpt2HelperFactory:
    @staticmethod
    def create_helper(helper_type="default"):
        helpers = {
            "default": Gpt2Helper,
            "beam_search_step": Gpt2BeamSearchHelper,
        }
        w = helpers[helper_type]
        return w

class GPT2LMHeadModel_BeamSearchStep(GPT2LMHeadModel):
    """Here we wrap a class for Onnx model conversion for GPT2LMHeadModel with past state and one 
    step beam search."""

    def __init__(self, 
                 config, 
                 batch_size, 
                 beam_size, 
                 temperature=1.0, 
                 repetition_penalty=1.0, 
                 excluded_token_ids=None, 
                 length_penalty=1.0, 
                 do_sample=False, 
                 do_sample_top_p=1, 
                 do_sample_top_k=0):
        super().__init__(config)
        self.config.batch_size = batch_size
        self.config.beam_size = beam_size
        self.config.temperature = temperature
        self.config.repetition_penalty = repetition_penalty
        self.config.excluded_token_ids = excluded_token_ids
        self.config.length_penalty = length_penalty
        self.config.do_sample = do_sample
        self.config.do_sample_top_p = do_sample_top_p
        self.config.do_sample_top_k = do_sample_top_k
    
    @staticmethod
    def collapse_first_two_dims(tensor):
        return tensor.view(-1, *tensor.size()[2:])
    
    @staticmethod
    def top_k_top_p_filtering(log_probs, top_p=1.0, top_k=0):
        '''Set tail event (out of top_p) to a big negative number'''
        sorted_log_probs, sorted_indices = torch.sort(log_probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_log_probs.exp(), dim=-1)
        sorted_indices_to_remove = cumulative_probs >= top_p
        sorted_indices_to_remove = torch.cat([torch.zeros_like(sorted_indices_to_remove[..., :1]), sorted_indices_to_remove[..., :-1]], dim=-1)
        if top_k > 0:
            sorted_indices_to_remove = torch.cat([sorted_indices_to_remove[..., :top_k], torch.ones_like(sorted_indices_to_remove[..., top_k:])], dim=-1)
        sorted_log_probs.masked_fill_(sorted_indices_to_remove, BIG_NEG)
        return log_probs.scatter(-1, sorted_indices, sorted_log_probs)

    def forward(
        self,
        input_ids,
        input_log_probs,
        input_unfinished_sents,
        prev_step_scores,
        *past,
    ):
        input_ids = input_ids.view(self.config.batch_size, -1, input_ids.size(-1))
        input_num_seq_per_sample = input_ids.size(1)

        input_ids_unfinished_flat = self.collapse_first_two_dims(input_ids).index_select(
            0, input_unfinished_sents.view(-1).nonzero(as_tuple=False).view(-1)
        )

        # attention_mask = (input_ids_unfinished_flat != self.config.eos_token_id).float()
        attention_mask = torch.ones(input_ids_unfinished_flat.shape).float().to(
            input_ids_unfinished_flat.device
        )
        position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0).long()

        if past:
            last_seq_len = past[0].size(-2)
            input_ids_unfinished_flat = input_ids_unfinished_flat[:, last_seq_len:]
            position_ids = position_ids[:, last_seq_len:]

        result = super().forward(
            input_ids_unfinished_flat.view(-1, input_ids_unfinished_flat.size(-1)),
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past,
            return_dict=False,
        )
        logits_flat, present_flat = MyGPT2Model.post_process(result, self.config.n_layer)

        # insert finished sequence back to form a square shape of (batch_size, beam_size)
        next_token_logits = logits_flat.new_zeros(input_ids.size()[:2] + (logits_flat.size(-1),))
        next_token_logits.index_fill_(2, torch.LongTensor([self.config.eos_token_id]).to(input_ids.device), -BIG_NEG)

        next_token_logits.masked_scatter_(input_unfinished_sents.unsqueeze(-1).expand_as(next_token_logits), logits_flat[:, -1])
        
        # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
        if self.config.repetition_penalty != 1.0:
            _pen = next_token_logits.gather(2, input_ids)
            _pen = torch.where(_pen > 0, _pen / self.config.repetition_penalty, _pen * self.config.repetition_penalty)
            next_token_logits.scatter_(2, input_ids, _pen)
        
        # similar way to encourage short sentence
        if self.config.length_penalty != 1.0:
            _pen = next_token_logits[..., self.config.eos_token_id]
            # if eos > 0, increase it, else, decrease it.
            _pen = torch.where(_pen > 0, _pen * self.config.length_penalty, _pen / self.config.length_penalty)
            next_token_logits[..., self.config.eos_token_id] = _pen
        
        if self.config.temperature != 1.0:
            next_token_logits = next_token_logits / self.config.temperature
        
        # exclude excluded_token_ids
        if self.config.excluded_token_ids is not None:
            next_token_logits.index_fill_(2, self.config.excluded_token_ids.to(next_token_logits.device), BIG_NEG)  # batch x beams/sequences x vocab_size

        next_token_log_probs = torch.log_softmax(next_token_logits, dim=-1)

        if self.config.do_sample:
            vocab_size = next_token_log_probs.size(-1)
            _next_token_log_probs = self.top_k_top_p_filtering(next_token_log_probs.view(-1, vocab_size), top_k=self.config.do_sample_top_k, top_p=self.config.do_sample_top_p)
            next_token_ids = torch.multinomial(_next_token_log_probs.exp(), num_samples=self.config.beam_size, replacement=False)
            next_token_ids = next_token_ids.view(self.config.batch_size, input_num_seq_per_sample, -1)
            next_token_log_probs = next_token_log_probs.gather(-1, next_token_ids)
        else:
            next_token_log_probs, next_token_ids = torch.topk(
                next_token_log_probs, self.config.beam_size, dim=-1, largest=True, sorted=True
            )

        output_log_probs = input_log_probs.unsqueeze(-1) + next_token_log_probs

        # select N sequences from beams of each input, sorted by sequence probability
        output_log_probs = output_log_probs.view(
            self.config.batch_size, -1
        )  # shape=(batch, beam_size^2)
        output_log_probs, selected_index_flat = output_log_probs.topk(
            self.config.beam_size, dim=-1, largest=True, sorted=True
        )  # output shape=(batch, beam_size)

        # select the correspondent sentences/next tokens
        selected_input_seq = selected_index_flat // self.config.beam_size
        next_token_ids = next_token_ids.view(self.config.batch_size, -1).gather(
            -1, selected_index_flat
        )

        prev_step_results = input_ids.view(
            self.config.batch_size, -1, input_ids.size(-1)
        ).contiguous()
        prev_step_results = prev_step_results.gather(1, selected_input_seq.unsqueeze(-1).expand(
            selected_input_seq.shape + (prev_step_results.size(-1),)
        ))

                
        output_unfinished_sents = input_unfinished_sents.gather(1, selected_input_seq)
        output_unfinished_sents = (
            output_unfinished_sents
            & next_token_ids.ne(self.config.eos_token_id)
        )

        current_step_results = torch.cat(
            [prev_step_results, next_token_ids.unsqueeze(-1)], dim=-1
        ).contiguous()

        prev_step_scores = prev_step_scores.view(
            self.config.batch_size, -1, prev_step_scores.size(-1)
        )
        prev_step_scores = prev_step_scores.gather(1, selected_input_seq.unsqueeze(-1).expand(
            selected_input_seq.shape + (prev_step_scores.size(-1),)
        ))
        current_step_scores = torch.cat(
            [prev_step_scores, output_log_probs.unsqueeze(-1)], dim=-1
        ).contiguous()

        # For next past state
        index_relative_to_last_unfinished = (
            input_unfinished_sents.view(-1).float().cumsum(-1) - 1
        ).clamp(min=0).long().reshape_as(input_unfinished_sents).gather(1, selected_input_seq)        
        unfinished_index_relative_to_last_unfinished = index_relative_to_last_unfinished.view(-1)[
            output_unfinished_sents.view(-1).nonzero(as_tuple=False).view(-1)
        ]

        present_flat = tuple(
            [p.index_select(1, unfinished_index_relative_to_last_unfinished) for p in present_flat]
        )

        return (
            current_step_results.view(self.config.batch_size * self.config.beam_size, -1),
            present_flat,
            output_log_probs,
            output_unfinished_sents,
            current_step_scores.view(self.config.batch_size * self.config.beam_size, -1),
        )


# Maps model class name to a tuple of model class, name of first output and use padding or not
MODEL_CLASSES = {
    'GPT2LMHeadModel': (MyGPT2LMHeadModel, 'logits', True),
    'GPT2LMHeadModel_NoPadding': (MyGPT2LMHeadModel_NoPadding, 'logits', False),
    'GPT2Model': (MyGPT2Model, 'last_state', True),
    "GPT2LMHeadModel_BeamSearchStep": (GPT2LMHeadModel_BeamSearchStep, "last_state", False), # defined in gpt2_beamsearch_helper.py
}


class Gpt2BeamSearchInputs(Gpt2Inputs):
    def __init__(
        self,
        input_ids,
        past,
        input_log_probs=None,
        input_unfinished_sents=None,
        prev_step_scores=None,
    ):
        super().__init__(input_ids, position_ids=None, attention_mask=None, past=past)
        self.prev_step_scores: Union[torch.FloatTensor, torch.HalfTensor, torch.cuda.FloatTensor] = prev_step_scores
        self.input_log_probs: Union[torch.FloatTensor, torch.HalfTensor, torch.cuda.FloatTensor] = input_log_probs
        self.input_unfinished_sents: torch.ByteTensor = input_unfinished_sents

    def to_list(self) -> List:
        input_list = [
            v 
            for v in [
                self.input_ids,
                self.input_log_probs, 
                self.input_unfinished_sents, 
                self.prev_step_scores
            ]
            if v is not None
        ]
        if self.past:
            input_list.extend(self.past)
        return input_list

    def to_fp32(self):
        past = [p.to(dtype=torch.float32) for p in self.past]
        return Gpt2BeamSearchInputs(
            self.input_ids,
            past,
            self.input_log_probs.to(dtype=torch.float32),
            self.input_unfinished_sents,
            self.prev_step_scores.to(dtype=torch.float32),
        )


class Gpt2BeamSearchHelper(Gpt2Helper):
    """A helper class for Gpt2 model conversion, inference and verification."""

    @staticmethod
    def get_dummy_inputs(batch_size: int,
                         past_sequence_length: int,
                         sequence_length: int,
                         num_attention_heads: int,
                         hidden_size: int,
                         num_layer: int,
                         vocab_size: int,
                         device: torch.device,
                         float16: bool = False,
                         has_position_ids: bool = True,
                         has_attention_mask: bool = True) -> Gpt2BeamSearchInputs:
        """Create random inputs for GPT2 model.
        Returns torch tensors of input_ids, position_ids, attention_mask and a list of past state tensors.
        """
        gpt2_dummy_inputs = Gpt2Helper.get_dummy_inputs(
            batch_size, 
            past_sequence_length, 
            sequence_length, 
            num_attention_heads,
            hidden_size,
            num_layer,
            vocab_size,
            device,
            float16,
            has_position_ids,
            has_attention_mask
        )
        float_type = torch.float16 if float16 else torch.float32

        input_log_probs = torch.zeros([batch_size, 1], dtype=float_type, device=device)
        input_unfinished_sents = torch.ones(
            [batch_size, 1], dtype=torch.bool, device=device
        )
        prev_step_scores = torch.zeros([batch_size, 1], dtype=float_type, device=device)

        return Gpt2BeamSearchInputs(
            gpt2_dummy_inputs.input_ids,
            gpt2_dummy_inputs.past,
            input_log_probs,
            input_unfinished_sents,
            prev_step_scores,
        )

    @staticmethod
    def get_output_shapes(batch_size: int,
                          context_len: int,
                          past_sequence_length: int,
                          sequence_length: int,
                          beam_size: int,
                          step: int,
                          config: GPT2Config,
                          model_class: str = "GPT2LMHeadModel",
                          num_seq: int = 0) -> Dict[str, List[int]]:
        """Returns a dictionary with output name as key, and shape as value."""
        num_attention_heads = config.num_attention_heads
        hidden_size = config.hidden_size
        num_layer = config.num_hidden_layers
        vocab_size = config.vocab_size       

        output_name = MODEL_CLASSES[model_class][1]

        last_state_shape = [batch_size * beam_size, past_sequence_length + sequence_length + 1]

        if num_seq == 0:
            num_seq = beam_size
        
        present_state_shape = [
            2,
            batch_size * num_seq,
            num_attention_heads,
            past_sequence_length + sequence_length,
            int(hidden_size / num_attention_heads),
        ]

        output_shapes = {output_name: last_state_shape}
        for i in range(num_layer):
            output_shapes["present_" + str(i)] = present_state_shape

        output_shapes["output_log_probs"] = [batch_size, beam_size]
        output_shapes["output_unfinished_sents"] = [batch_size, beam_size]
        output_shapes["current_step_scores"] = [batch_size * beam_size, past_sequence_length + sequence_length - context_len + 2]
        return output_shapes

    @staticmethod
    def get_output_buffers(
        output_shapes, device, is_float16=False
    ):
        """Returns a dictionary of output name as key, and 1D tensor as value. The tensor has enough space for given shape."""
        data_type = torch.float16 if is_float16 else torch.float32

        output_buffers = {}
        for name, shape in output_shapes.items():
            if name == "last_state":
                output_buffers[name] = torch.empty(
                    numpy.prod(shape), dtype=torch.long, device=device
                )
            elif name == "output_unfinished_sents":
                output_buffers[name] = torch.empty(
                    numpy.prod(shape), dtype=torch.bool, device=device
                )
            else:
                output_buffers[name] = torch.empty(
                    numpy.prod(shape), dtype=data_type, device=device
                )
        return output_buffers

    @staticmethod
    def compare_outputs(torch_outputs, ort_outputs, rtol=1e-03, atol=1e-03):
        """Returns True if torch and ORT outputs are close for given thresholds, and False otherwise."""
        is_close = numpy.allclose(
            ort_outputs[0], torch_outputs[0].cpu().numpy(), rtol=rtol, atol=atol
        )
        logger.debug(
            f"PyTorch and OnnxRuntime output 0 (last_state) are close: {is_close}"
        )

        if not is_close:
            max_abs_diff = Gpt2BeamSearchHelper.diff_outputs(torch_outputs, ort_outputs)
            logger.info(
                f"PyTorch and OnnxRuntime results are not all close: max_abs_diff={max_abs_diff:.5f}"
            )

        return is_close

    @staticmethod
    def export_onnx(model,
                    device,
                    onnx_model_path: str,
                    verbose: bool = False,
                    use_external_data_format: bool = False,
                    has_position_ids: bool = True,
                    has_attention_mask: bool = True):
        """Export GPT-2 model with past state to ONNX model."""
        config: GPT2Config = model.config
        num_layer = config.n_layer
        dummy_inputs = Gpt2BeamSearchHelper.get_dummy_inputs(batch_size=1,
                                                             past_sequence_length=1,
                                                             sequence_length=2,
                                                             num_attention_heads=config.num_attention_heads,
                                                             hidden_size=config.hidden_size,
                                                             num_layer=num_layer,
                                                             vocab_size=config.vocab_size,
                                                             device=device,
                                                             float16=False,
                                                             has_position_ids=has_position_ids,
                                                             has_attention_mask=has_attention_mask)
        input_list = dummy_inputs.to_list()

        with torch.no_grad():
            outputs = model(*input_list)

        past_names = [f"past_{i}" for i in range(num_layer)]
        present_names = [f"present_{i}" for i in range(num_layer)]

        output_names = ["last_state"] + present_names

        output_names += [
            "output_log_probs",
            "output_unfinished_sents",
            "current_step_scores",
        ]

        # Shape of input tensors:
        #    input_ids: (batch_size, seq_len)
        #    past_{i}:  (2, batch_size, num_heads, past_seq_len, hidden_size/num_heads)
        #    attention_mask: (batch_size, past_seq_len + seq_len)
        # Shape of output tensors:
        #    last_state: (batch_size, seq_len, hidden_size)
        #      or logits: (batch_size, seq_len, vocab_size)
        #    present_{i}:  (2, batch_size, num_heads, past_seq_len + seq_len, hidden_size/num_heads)
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "cur_seq_len"},
            output_names[0]: {0: "batch_size", 1: "next_seq_len"},
        }
        for name in past_names:
            dynamic_axes[name] = {1: "batch_size", 3: "past_seq_len"}
        for name in present_names:
            dynamic_axes[name] = {1: "batch_size", 3: "cur_seq_len"}

        input_names = ["input_ids"]
        dynamic_axes["input_log_probs"] = {0: "batch_size", 1: "beam_size"}
        input_names.append("input_log_probs")
        dynamic_axes["input_unfinished_sents"] = {0: "batch_size", 1: "beam_size"}
        input_names.append("input_unfinished_sents")
        dynamic_axes["prev_step_scores"] = {0: "batch_size", 1: "total_seq_len"}
        input_names.append("prev_step_scores")
        input_names.extend(past_names)

        logger.info(
            f"Shapes: input_ids={dummy_inputs.input_ids.shape} past={dummy_inputs.past[0].shape} output={outputs[0].shape} present={outputs[1][0].shape}"
        )

        Path(onnx_model_path).parent.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            model,
            args=tuple(input_list),
            f=onnx_model_path,
            input_names=input_names,
            output_names=output_names,
            example_outputs=outputs,
            dynamic_axes=dynamic_axes,
            opset_version=12,
            do_constant_folding=True,
            use_external_data_format=use_external_data_format,
            verbose=verbose,
        )

    @staticmethod
    def onnxruntime_inference(ort_session, inputs: Gpt2BeamSearchInputs, total_runs: int = 0):
        """Run inference of ONNX model, and returns average latency in ms when total_runs > 0 besides outputs."""
        logger.debug(f"start onnxruntime_inference")

        ort_inputs = {
            "input_ids": numpy.ascontiguousarray(inputs.input_ids.cpu().numpy())
        }

        if inputs.input_log_probs is not None:
            ort_inputs["input_log_probs"] = numpy.ascontiguousarray(
                inputs.input_log_probs.cpu().numpy()
            )
        if inputs.input_unfinished_sents is not None:
            ort_inputs["input_unfinished_sents"] = numpy.ascontiguousarray(
                inputs.input_unfinished_sents.cpu().numpy()
            )
        if inputs.prev_step_scores is not None:
            ort_inputs["prev_step_scores"] = numpy.ascontiguousarray(
                inputs.prev_step_scores.cpu().numpy()
            )
        if inputs.past is not None:
            for i, past_i in enumerate(inputs.past):
                ort_inputs[f"past_{i}"] = numpy.ascontiguousarray(past_i.cpu().numpy())

        ort_outputs = ort_session.run(None, ort_inputs)
        if total_runs == 0:
            return ort_outputs

        latency = []
        for _ in range(total_runs):
            start = time.time()
            ort_outputs = ort_session.run(None, ort_inputs)
            latency.append(time.time() - start)

        average_latency = sum(latency) * 1000 / len(latency)
        logger.debug(
            "OnnxRuntime Inference time = {} ms".format(format(average_latency, ".2f"))
        )

        return ort_outputs, average_latency

    @staticmethod
    def prepare_io_binding(ort_session,
                           input_ids,
                           past,
                           output_buffers,
                           output_shapes,
                           input_log_probs=None,
                           input_unfinished_sents=None,
                           prev_step_scores=None):
        """Returnas IO binding object for a session."""

        # Bind inputs and outputs to onnxruntime session
        io_binding = Gpt2Helper.prepare_io_binding(ort_session, 
                                                   input_ids, 
                                                   position_ids=None, 
                                                   attention_mask=None, 
                                                   past=past, 
                                                   output_buffers=output_buffers, 
                                                   output_shapes=output_shapes)

        # Bind inputs
        data_type = output_buffers[ort_session.get_outputs()[1].name].dtype
        float_type = numpy.float16 if data_type == torch.float16 else numpy.float32

        if past is not None:
            for i, past_i in enumerate(past):
                assert past_i.is_contiguous()

                data_ptr = past_i.data_ptr()
                if data_ptr == 0:
                    # When past_sequence_length is 0, its data_ptr will be zero. IO Binding asserts that data_ptr shall not be zero.
                    # Here we workaround and pass data pointer of input_ids. Actual data is not used for past so it does not matter.
                    data_ptr = input_ids.data_ptr()

                io_binding.bind_input(f'past_{i}', past_i.device.type, 0, float_type, list(past_i.size()), data_ptr)

        if input_log_probs is not None:
            assert input_log_probs.is_contiguous()
            io_binding.bind_input(
                "input_log_probs",
                input_log_probs.device.type,
                0,
                float_type,
                list(input_log_probs.size()),
                input_log_probs.data_ptr(),
            )

        if input_unfinished_sents is not None:
            assert input_unfinished_sents.is_contiguous()
            io_binding.bind_input(
                "input_unfinished_sents",
                input_unfinished_sents.device.type,
                0,
                numpy.bool,
                list(input_unfinished_sents.size()),
                input_unfinished_sents.data_ptr(),
            )

        if prev_step_scores is not None:
            assert prev_step_scores.is_contiguous()
            io_binding.bind_input(
                "prev_step_scores",
                prev_step_scores.device.type,
                0,
                float_type,
                list(prev_step_scores.size()),
                prev_step_scores.data_ptr(),
            )

        # Bind outputs
        for output in ort_session.get_outputs():
            output_name = output.name
            output_buffer = output_buffers[output_name]
            logger.debug(
                f"{output_name} device type={output_buffer.device.type} shape={list(output_buffer.size())}"
            )
            if output_name == "last_state":
                io_binding.bind_output(
                    output_name,
                    output_buffer.device.type,
                    0,
                    numpy.longlong,
                    output_shapes[output_name],
                    output_buffer.data_ptr(),
                )
            elif output_name == "output_unfinished_sents":
                io_binding.bind_output(
                    output_name,
                    output_buffer.device.type,
                    0,
                    numpy.bool,
                    output_shapes[output_name],
                    output_buffer.data_ptr(),
                )
            else:
                io_binding.bind_output(
                    output_name,
                    output_buffer.device.type,
                    0,
                    float_type,
                    output_shapes[output_name],
                    output_buffer.data_ptr(),
                )

        return io_binding

    @staticmethod
    def onnxruntime_inference_with_binded_io(ort_session,
                                             inputs: Gpt2BeamSearchInputs,
                                             output_buffers: Dict[str, torch.Tensor],
                                             output_shapes: Dict[str, List[int]],
                                             total_runs: int = 0,
                                             return_numpy: bool = True,
                                             include_copy_output_latency: bool = False):
        """Inference with IO binding. Returns outputs, and optional latency when total_runs > 0.
        """
        logger.debug(f"start onnxruntime_inference_with_binded_io")

        # Bind inputs and outputs to onnxruntime session
        io_binding = Gpt2BeamSearchHelper.prepare_io_binding(
            ort_session,
            inputs.input_ids,
            inputs.past,
            output_buffers,
            output_shapes,
            inputs.input_log_probs,
            inputs.input_unfinished_sents,
            inputs.prev_step_scores,
        )

        # Run onnxruntime with io binding
        ort_session.run_with_iobinding(io_binding)

        # Copy results to cpu for verification
        ort_outputs = Gpt2BeamSearchHelper.get_outputs_from_io_binding_buffer(
            ort_session, output_buffers, output_shapes, return_numpy
        )

        if total_runs == 0:
            return ort_outputs

        latency = []
        for _ in range(total_runs):
            start = time.time()
            # Run onnxruntime with io binding
            ort_session.run_with_iobinding(io_binding)
            if include_copy_output_latency:
                _ = Gpt2BeamSearchHelper.get_outputs_from_io_binding_buffer(
                    ort_session, output_buffers, output_shapes, return_numpy
                )
            latency.append(time.time() - start)

        average_latency = sum(latency) * 1000 / len(latency)
        logger.debug(
            "OnnxRuntime with IO binding inference time = {} ms".format(
                format(average_latency, ".2f")
            )
        )

        return ort_outputs, average_latency

    @staticmethod
    def test_parity(ort_session,
                    model,
                    device,
                    is_float16=False,
                    rtol=5e-4,
                    atol=5e-4,
                    total_test_cases=100,
                    use_io_binding=True,
                    model_class="GPT2LMHeadModel_BeamSearchStep",
                    has_position_ids=True,
                    has_attention_mask=True):
        """Generate random inputs and compare the results of PyTorch and Onnx Runtime."""

        config: GPT2Config = model.config

        logger.info(
            f"Running parity test (rtol={rtol}, atol={atol}, test_cases={total_test_cases}, use_io_binding={use_io_binding} model_class={model_class} is_float16={is_float16}) ..."
        )

        max_batch_size = 1
        max_past_seq_len = 4  # Do not use large number here for higher chance of hitting empty past (past_seq_len=0)
        max_seq_len = 2
        beam_size = 4

        output_buffers = None
        if use_io_binding:
            max_output_shapes = Gpt2BeamSearchHelper.get_output_shapes(
                max_batch_size,
                max_past_seq_len,
                max_past_seq_len,
                max_seq_len,
                beam_size,
                0,
                config,
                model_class,
            )
            output_buffers = Gpt2BeamSearchHelper.get_output_buffers(
                max_output_shapes, device, is_float16
            )

        passed_test_cases = 0
        for _ in range(total_test_cases):
            past_sequence_length = random.randint(0, max_past_seq_len)
            sequence_length = random.randint(1 + past_sequence_length, max_seq_len + past_sequence_length)
            batch_size = random.randint(1, max_batch_size)

            logger.debug(
                f"Running parity test for batch_size={batch_size} past_sequence_length={past_sequence_length}..."
            )
            dummy_inputs = Gpt2BeamSearchHelper.get_dummy_inputs(
                batch_size,
                past_sequence_length,
                sequence_length,
                config.num_attention_heads,
                config.hidden_size,
                config.n_layer,
                config.vocab_size,
                device,
                is_float16, 
                has_position_ids,
                has_attention_mask
            )

            outputs = Gpt2BeamSearchHelper.pytorch_inference(model, dummy_inputs)
            if use_io_binding:
                ort_outputs = Gpt2BeamSearchHelper.onnxruntime_inference(
                    ort_session, dummy_inputs
                )
            else:
                output_shapes = Gpt2BeamSearchHelper.get_output_shapes(
                    batch_size,
                    past_sequence_length,
                    past_sequence_length,
                    sequence_length,
                    beam_size,
                    0,
                    config,
                    model_class,
                )
                ort_outputs = Gpt2BeamSearchHelper.onnxruntime_inference_with_binded_io(
                    ort_session, dummy_inputs, output_buffers, output_shapes
                )

            is_all_close = Gpt2BeamSearchHelper.compare_outputs(
                outputs, ort_outputs, rtol=rtol, atol=atol
            )
            if is_all_close:
                passed_test_cases += 1
        logger.info(f"Parity Test Cases={total_test_cases}; Passed={passed_test_cases}")
        if passed_test_cases > 0.95 * total_test_cases:
            logger.info(
                f"Parity is good: passed rate={int(passed_test_cases*100/total_test_cases):.0f}%"
            )
        return passed_test_cases == total_test_cases

    @staticmethod
    def torchscript(model, config, device, has_position_ids=True, has_attention_mask=True):
        """JIT trace for TorchScript."""
        input_list = Gpt2BeamSearchHelper.get_dummy_inputs(
            batch_size=1,
            past_sequence_length=1,
            sequence_length=1,
            num_attention_heads=config.num_attention_heads,
            hidden_size=config.hidden_size,
            num_layer=config.n_layer,
            vocab_size=config.vocab_size,
            device=device,
            float16=False,
            has_position_ids=has_position_ids,
            has_attention_mask=has_attention_mask,
        ).to_list()
        return torch.jit.trace(model, input_list)
