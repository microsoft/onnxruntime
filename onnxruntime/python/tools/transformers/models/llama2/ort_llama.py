import time
import torch

import numpy as np
import onnxruntime as ort

from torch import nn

# from onnxruntime.training.ortmodule._utils import _ortvalues_to_torch_tensor
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


ORT_TYPE_TO_NP_TYPE = {
    "tensor(float16)": np.float16,
    "tensor(float)": np.float32,
    "tensor(int8)": np.int8,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(bool)": bool,
}


class OrtModelForLlamaCausalLM(PreTrainedModel):
    def __init__(self, args, model_f, local_rank, sess_opt, provider_opt, **kwargs):
        super().__init__(**kwargs)
        self.main_input_name = "input_ids"
        self.one = nn.Parameter(torch.tensor([0]), requires_grad=False)

        self.sess = None
        self.sess_with_past = None
        self.merged_sess = None

        if args.provider == "rocm":
            ep = "ROCMExecutionProvider"
        elif args.provider == "cuda":
            ep = "CUDAExecutionProvider"
        else:
            raise ValueError(f"Unknown provider {args.provider}")

        def _get_past_k_logits(session):
            past_k = next((i for i in session.get_inputs() if i.name == "past_key_values.0.key"), None)
            logits = next((i for i in session.get_outputs() if i.name == "logits"), None)
            return past_k, logits

        if args.merge:
            self.merged_sess = ort.InferenceSession(model_f, sess_options=sess_opt, providers=[(ep, provider_opt)])
            past_k, logits = _get_past_k_logits(self.merged_sess)
        else:
            decoder_model, decoder_past_model = model_f
            self.sess = ort.InferenceSession(decoder_model, sess_options=sess_opt, providers=[(ep, provider_opt)])

            if sess_opt.enable_profiling:
                sess_opt.profile_file_prefix = sess_opt.profile_file_prefix + "_past"
            self.sess_with_past = ort.InferenceSession(
                decoder_past_model, sess_options=sess_opt, providers=[(ep, provider_opt)]
            )
            past_k, logits = _get_past_k_logits(self.sess_with_past)

        config = kwargs.get("config", None)
        assert config is not None, "need input config for OrtModelForLlamaCausalLM"
        self.n_layers = config.num_hidden_layers
        self.num_heads = past_k.shape[1]
        self.head_dim = past_k.shape[3]
        self.vocab_size = config.vocab_size
        self.torch_dtype = config.torch_dtype
        self.rank = local_rank
        self.cost = 0
        self.skip_warm = 2
        self.iters = 0

    def can_generate(self):
        return True

    def forward_with_io_binding(self, input_ids, attn_mask, position_ids, past_kvs=None):
        sess = (
            self.merged_sess
            if self.merged_sess is not None
            else (self.sess if past_kvs is None else self.sess_with_past)
        )
        name_map = {"input_ids": input_ids, "attention_mask": attn_mask, "position_ids": position_ids}

        if self.merged_sess is not None:
            name_map["use_cache_branch"] = torch.tensor([past_kvs is not None])

        if past_kvs != None:
            assert len(past_kvs) == self.n_layers * 2

        for i in range(self.n_layers):
            name_map[f"past_key_values.{i}.key"] = (
                past_kvs[2 * i]
                if past_kvs is not None
                else torch.empty((1, self.num_heads, 1, self.head_dim), device=self.device)
            )
            name_map[f"past_key_values.{i}.value"] = (
                past_kvs[2 * i + 1]
                if past_kvs is not None
                else torch.empty((1, self.num_heads, 1, self.head_dim), device=self.device)
            )

        io_binding = sess.io_binding()
        for i in sess.get_inputs():
            t = name_map[i.name]
            if i.name == "use_cache_branch":
                io_binding.bind_input(i.name, "cpu", 0, ORT_TYPE_TO_NP_TYPE[i.type], tuple(t.shape), t.data_ptr())
            else:
                io_binding.bind_input(
                    i.name, "cuda", self.rank, ORT_TYPE_TO_NP_TYPE[i.type], tuple(t.shape), t.data_ptr()
                )

        outputs = []
        output = torch.empty((*input_ids.shape, self.vocab_size), dtype=torch.float32, device=self.device)
        seq_len = input_ids.shape[-1]
        if past_kvs is not None:
            seq_len += past_kvs[0].shape[2]

        name_map["logits"] = output
        outputs.append(output)

        present_kv_shape = (input_ids.shape[0], self.num_heads, seq_len, self.head_dim)
        for i in range(self.n_layers):
            k = torch.empty(present_kv_shape, dtype=self.torch_dtype, device=self.device)
            name_map[f"present.{i}.key"] = k
            outputs.append(k)
            v = torch.empty(present_kv_shape, dtype=self.torch_dtype, device=self.device)
            name_map[f"present.{i}.value"] = v
            outputs.append(v)

        for out in sess.get_outputs():
            t = name_map[out.name]
            io_binding.bind_output(
                out.name,
                "cuda",
                self.rank,
                ORT_TYPE_TO_NP_TYPE[out.type],
                tuple(t.shape),
                t.data_ptr(),
            )

        start = time.time()
        io_binding.synchronize_inputs()
        sess.run_with_iobinding(io_binding)
        io_binding.synchronize_outputs()
        if past_kvs is not None:
            self.iters += 1
            if self.iters >= self.skip_warm:
                self.cost += time.time() - start

        # outputs = io_binding.get_outputs_as_ortvaluevector()
        # outputs = _ortvalues_to_torch_tensor(outputs)
        return outputs

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, past_key_values=None, **kwargs):
        if past_key_values:
            input_ids = input_ids[:, -1:]
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        model_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
        }

        return model_inputs

    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, **kwargs):
        results = self.forward_with_io_binding(input_ids, attention_mask, position_ids, past_key_values)
        logits, past_key_values = results[0], results[1:]

        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)
