# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import numpy as np
import torch
from transformers import AutoTokenizer

import onnxruntime as ort

pt_to_np = {
    "torch.int32": np.int32,
    "torch.int64": np.int64,
    "torch.float32": np.float32,
    "torch.float16": np.float16,
}


class ORTGenerator:
    def __init__(self, decoder_path):
        self.onnx_decoder_path = decoder_path
        self.num_heads = 32
        self.head_size = 80
        self.num_layers = 32
        self.max_sequence_length = 2048

    def get_initial_inputs_and_outputs(self, encodings_dict):
        self.torch_dtype = torch.float16 if self.use_fp16 else torch.float32

        input_ids = torch.tensor(encodings_dict["input_ids"], device=self.device, dtype=torch.int32)
        attention_mask = torch.tensor(encodings_dict["attention_mask"], device=self.device, dtype=torch.int32)
        step = torch.tensor([0], device=self.device, dtype=torch.int64)

        inputs = {
            "input_ids": input_ids.contiguous(),
            "attention_mask": attention_mask.contiguous(),
        }

        if self.use_step:
            inputs["step"] = step.contiguous()

        batch_size, sequence_length = input_ids.shape

        past_seq_length = self.max_sequence_length if self.use_buffer_share else 0
        past_shape = (
            (2, batch_size, self.num_heads, past_seq_length, self.head_size)
            if self.packed_kv
            else (batch_size, self.num_heads, past_seq_length, self.head_size)
        )
        for i in range(self.num_layers):
            past = torch.zeros(past_shape, device=self.device, dtype=self.torch_dtype)
            inputs.update(
                {f"past_key_{i}": past.contiguous(), f"past_value_{i}": past.clone().contiguous()}
            ) if not self.packed_kv else inputs.update({f"past_{i}": past.contiguous()})

        logits = torch.zeros(batch_size, sequence_length, 51200, device=self.device, dtype=self.torch_dtype)
        outputs = {"logits": logits.contiguous()}

        if not self.use_buffer_share:
            present_shape = (
                (2, batch_size, self.num_heads, sequence_length, self.head_size)
                if self.packed_kv
                else (batch_size, self.num_heads, sequence_length, self.head_size)
            )
            for i in range(self.num_layers):
                present = torch.zeros(present_shape, device=self.device, dtype=self.torch_dtype)
                outputs.update(
                    {f"present_key_{i}": present.contiguous(), f"present_value_{i}": present.contiguous()}
                ) if not self.packed_kv else outputs.update({f"present_{i}": present.contiguous()})

        return inputs, outputs

    def apply_io_binding(self, model: ort.InferenceSession, inputs: dict, outputs: dict):
        io_binding = model.io_binding()
        device = None

        for k, v in inputs.items():
            io_binding.bind_input(
                name=k,
                device_type=v.device.type,
                device_id=0 if v.device.type == "cpu" else v.device.index,
                element_type=pt_to_np[repr(v.dtype)],
                shape=tuple(v.shape),
                buffer_ptr=v.data_ptr(),
            )
            device = v.device

        for output in model.get_outputs():
            name = output.name
            if self.use_buffer_share and "present" in name:
                v = inputs[name.replace("present", "past")]
                io_binding.bind_output(
                    name=name,
                    device_type=v.device.type,
                    device_id=v.device.index,
                    element_type=(np.float16 if self.use_fp16 else np.float32),
                    shape=tuple(v.shape),
                    buffer_ptr=v.data_ptr(),
                )
            else:
                v = outputs[name]
                io_binding.bind_output(
                    name=name,
                    device_type=device.type,
                    device_id=0 if device.type == "cpu" else device.index,
                    element_type=(np.float16 if self.use_fp16 else np.float32),
                    shape=tuple(v.shape),
                    buffer_ptr=v.data_ptr(),
                )

        return io_binding

    def create_session(self, device_id, use_fp16=True, use_buffer_share=True, packed_kv=False, use_step=False):
        sess_options = ort.SessionOptions()
        ep = ("CUDAExecutionProvider", {"device_id": device_id}) if device_id >= 0 else "CPUExecutionProvider"
        self.sess = ort.InferenceSession(self.onnx_decoder_path, sess_options=sess_options, providers=[ep])

        self.device = torch.device("cuda", device_id) if torch.cuda.is_available() else torch.device("cpu")
        self.use_fp16 = use_fp16
        self.use_buffer_share = use_buffer_share
        self.packed_kv = packed_kv
        self.use_step = use_step

        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        self.tokenizer.pad_token = "[PAD]"

    def generate(self, prompt, max_length):
        encodings_dict = self.tokenizer.batch_encode_plus(prompt, padding=True)

        inputs, outputs = self.get_initial_inputs_and_outputs(encodings_dict)

        all_token_ids = inputs["input_ids"].clone()
        batch_size, sequence_length = all_token_ids.shape

        current_length = sequence_length
        has_eos = torch.zeros(batch_size, device=self.device, dtype=torch.bool)

        while current_length < max_length:
            io_binding = self.apply_io_binding(self.sess, inputs, outputs)

            io_binding.synchronize_inputs()
            self.sess.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()

            # Sample with argmax (greedy search)
            next_token_logits = outputs["logits"][:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1)

            # Check if we previously reached EOS token id or if generated token id is EOS token id
            has_eos = has_eos | next_tokens == self.tokenizer.eos_token_id

            # Determine which new tokens to add to list of all token ids
            # Add EOS token ids for batch entries that ended early (ragged batching scenario where some batch entries ended early and some haven't)
            tokens_to_add = next_tokens.masked_fill(has_eos, self.tokenizer.eos_token_id).reshape([batch_size, 1])
            all_token_ids = torch.cat([all_token_ids, tokens_to_add], dim=-1)

            # Return early if all batch entries have reached EOS token id
            if torch.all(has_eos):
                break

            # Update inputs for next inference run
            current_length += 1
            inputs["input_ids"] = tokens_to_add.to(torch.int32)
            if self.use_step:
                inputs["step"] = torch.tensor([current_length - 1], device=self.device, dtype=torch.int64)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], (~has_eos).reshape(batch_size, 1)], 1).to(
                torch.int32
            )

            # Set logits to zeros for next inference run and re-use memory buffer
            if outputs["logits"].shape[1] != 1:
                outputs["logits"] = outputs["logits"][:, :1, :].contiguous()
            outputs["logits"].zero_()

            if not self.use_buffer_share:
                for i in range(self.num_layers):
                    if not self.packed_kv:
                        inputs[f"past_key_{i}"] = outputs[f"present_key_{i}"]
                        inputs[f"past_value_{i}"] = outputs[f"present_value_{i}"]
                    else:
                        inputs[f"past_{i}"] = outputs[f"present_{i}"]

                new_sequence_length = inputs["attention_mask"].shape[1]
                present_shape = (
                    (2, batch_size, self.num_heads, new_sequence_length, self.head_size)
                    if self.packed_kv
                    else (batch_size, self.num_heads, new_sequence_length, self.head_size)
                )
                for i in range(self.num_layers):
                    present = torch.zeros(present_shape, device=self.device, dtype=self.torch_dtype)
                    outputs.update(
                        {f"present_key_{i}": present.contiguous(), f"present_value_{i}": present.clone().contiguous()}
                    ) if not self.packed_kv else outputs.update({f"present_{i}": present.contiguous()})

        texts = self.tokenizer.batch_decode(all_token_ids, skip_special_tokens=True)
        return texts


def run_phi2(onnx_model_path, use_buffer_share, device_id, packed_kv=False, use_fp16=True, use_step=False):
    prompt = [
        '''```python
    def print_prime(n):
    """
    Print all primes between 1 and n
    """'''
    ]

    generator = ORTGenerator(onnx_model_path)
    generator.create_session(device_id, use_fp16, use_buffer_share, packed_kv, use_step)
    texts = generator.generate(prompt, max_length=200)

    for i in range(len(texts)):
        print("Prompt: ", prompt[i])
        print("Texts: ", texts[i])
