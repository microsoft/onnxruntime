---
title: Config reference
description: Reference for the ONNX Runtime Generative AI configuration file
has_children: false
parent: Reference
grand_parent: Generative AI (Preview)
nav_order: 1
---

# Configuration reference 

_Note: this API is in preview and is subject to change._

A configuration file called genai_config.json is generated automatically if the model is generated with the model builder. If you provide your own model, you can copy the example below and modify it for your scenario.


## Example file for phi-2

```json
{
    "model": {
        "bos_token_id": 50256,
        "context_length": 2048,
        "decoder": {
            "session_options": {
                "log_id": "onnxruntime-genai",
                "provider_options": [
                    {
                        "cuda": {}
                    }
                ]
            },
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
        "do_sample": false,
        "early_stopping": true,
        "length_penalty": 1.0,
        "max_length": 20,
        "min_length": 0,
        "no_repeat_ngram_size": 0,
        "num_beams": 1,
        "num_return_sequences": 1,
        "past_present_share_buffer": true,
        "repetition_penalty": 1.0,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 1.0
    }
}
```

## Configuration

### Model section

#### General model config

* _type_: The type of model. Can be phi, llama or gpt.

* _vocab_size_: The size of the vocabulary that the model processes ie the number of tokens in the vocabulary.

* _bos_token_id_: The id of the beginning of sequence token.

* _eos_token_id_: The id of the end of sequence token.

* _pad_token_: The id of the padding token.

* _context_length_: The maxinum length of sequence that the model can process.

* _pad_token_: The id of the padding token.

#### Session options

These are the options that are passed to ONNX Runtime, which runs the model on each token generation iteration.

* _provider_options_: a priortized list of execution targets on which to run the model. If running on CPU, this option is not present. A list of execution provider specific configurations can be specified inside the provider item.

* _log_id_: a prefix to output when logging


Then For each model in the pipeline there is one section, named by the model. 

#### Decoder model config

* _filename_: The name of the model file.

* _inputs_: The names of each of the inputs. Sequences of model inputs can contain a wildcard representing the index in the sequence.

* _outputs_: The names of each of the outputs.

* _num_attention_heads: The number of attention heads in the model.

* _head_size_: The size of the attention heads.

* _hidden_size_: The size of the hidden layers.

* _num_key_value_heads_: The number of key value heads.


### Search section

* _max_length_: The maximum length that the model will generate.
* _min_length_: The minimum length that the model will generate.
* _do_sample_: 
* _num_beams_: The number of beams to apply when generating the output sequence using beam search.
* _num_sequences_: The number of sequences to generate. Returns the sequences with the highest scores in order.
words are repeated.
* _temperature_: 
* _top_k_:
* _top_p_:
* _early_stopping_ :
* _repetition_penalty_: The penalty to apply when 
* _length_penalty_: 
* _diversity_penalty_: 
* _no_repeat_ngram_size_: 
* _past_present_share_buffer_: 
