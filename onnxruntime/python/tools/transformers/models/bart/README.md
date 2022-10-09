# Bart ONNX Exporter with Custom Ops Beam Search

This folder exports BART model from huggingface to ONNX, specifically on `BartForConditionalGeneration`

## prerequisites

- 3.6<=Python<3.10
- Pytorch nightly
- onnx nightly or >= 1.12
- onnxruntime nightly
- transformers nightly

## Usage

  - [Steps](#all-steps)
    - [Encoder](#encoder)
    - [decoder](#decoder)
    - [Combine and Convert](#combine-and-convert)
    - [inference](#inference)
  - [Required Parameters](#required-parameters)
  - [Optional Parameters](#optional-parameters)

### Introduction

We break BART into 3 parts: encoder, decoder, and beam search. As long as you confirm
that you have the certain part of the model, you are allowed to skip it with parameter.

1. no_encoder (skip converting encoder)
2. no_decoder (skip converting decoder)
3. no_chain (skip combining encoder/decoder with beam search)
4. no_inference (skip final inference)

### All Steps

Four steps are included, but they are not necessarily to be activated at the same time.
There are encoder, decoder, combine and convert, and inference.

```bash
python export.py -m facebook/bart-base
```

---

#### Encoder

Encoder and its init generator generates encoder and its init wrapper into one ONNX model. Notice that official model is supported, if an user doesn't have a local model to provide.

```bash
python export.py -m facebook/bart-base --no_decoder --no_chain --no_inference
```

Notice that model will be /path/to/output/edinit.onnx

---

#### Decoder

Decoder generator generates decoder ONNX model.

```bash
python export.py -m facebook/bart-base --no_encoder --no_chain --no_inference
```

Notice that model will be /path/to/output/decoder.onnx

---

#### Combine and Convert

This is combining the model with custom operator and export to a final ONNX model. Notice that you should have encoder and decoder ONNX model before running this.

```bash
python export.py -m facebook/bart-base --no_encoder --no_decoder --no_inference
```

Notice that model will be /path/to/output/model_final.onnx

---

#### Inference

Inferencing with pytorch model and `final ONNX model` we generated. Notice that you should have final ONNX model before running this.

```bash
python export.py -m facebook/bart-base --no_encoder --no_decoder --no_chain
```

---

### Required Parameters

| Name | flag | type | Description |
| --- | --- | --- | --- |
| --model_dir | -m | string | pytorch model directory, and also output model directory |

### Optional Parameters

| Name | flag | type | Description |
| --- | --- | --- | --- |
| --max_length | N/A | int | Default to 256. maximum length of generated sequence. |
| --min_length | N/A | int | Default to 20. minimum length of generated sequence. |
| --output | -o | string | default name is onnx_models. output directory under model_dir |
| --input_text | -i | string | input a paragraph of text or use default text in export.py |
| --num_beams | -b | int | default to 5 |
| --spm_path | -s | string | tokenizer model from sentencepice. Use huggingface tokenizer if this is not provided |
| --repetition_penalty | N/A | int | default to 1 |
| --no_repeat_ngram_size | N/A | int | default to 3 |
| --opset_version | N/A | int | default to 14 |
