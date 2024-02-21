---
title: Configuration reference
description: Reference for the ONNX Runtime Generative AI configuration file
has_children: false
parent: Reference
grand_parent: Generative AI (Preview)
nav_order: 1
---

# Configuration reference 

_Note: this API is in preview and is subject to change._


## Example file for phi-2

```
{
    "model": {
        "bos_token_id": 50256,
        "context_length": 2048,
        "decoder": {
            "filename": "model.onnx",
            "head_size": 80,
            "hidden_size": 2560,
            "inputs": {
                "input_ids": "input_ids",
                "attention_mask": "attention_mask",
                "position_ids": "position_ids",
                "past_key_names": "past_key_values.%d.key",
                "past_value_names": "past_key_values.%d.value"
            },
            "outputs": {
                "logits": "logits",
                "present_key_names": "present.%d.key",
                "present_value_names": "present.%d.value"
            },
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "num_key_value_heads": 32
        },
        "eos_token_id": 50256,
        "pad_token_id": 50256,
        "type": "phi",
        "vocab_size": 51200
    },
    "search": {
        "diversity_penalty": 0.0,
        "length_penalty": 1.0,
        "max_length": 20,
        "min_length": 0,
        "no_repeat_ngram_size": 0,
        "num_beams": 1,
        "num_return_sequences": 1,
        "repetition_penalty": 1.0,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.6
    }
}
```
