# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import logging
import numpy
import os
import torch
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer, AutoModel
from benchmark_helper import create_onnxruntime_session, Precision
from gpt2_helper import GPT2ModelNoPastState, PRETRAINED_GPT2_MODELS
from quantize_helper import QuantizeHelper
from huggingface_models import MODEL_CLASSES

logger = logging.getLogger(__name__)

# Walkaround by replacing torch.triu using self-defined op
# Since torch.triu cannot be exported to ONNX. See https://github.com/pytorch/pytorch/issues/32968
torch_func = {"triu": torch.triu}


def triu_onnx(x, diagonal=0, out=None):
    assert out is None
    assert len(x.shape) == 2 and x.size(0) == x.size(1)

    torch_triu = torch_func["triu"]
    template = torch_triu(torch.ones((1024, 1024), dtype=torch.uint8), diagonal)
    mask = template[:x.size(0), :x.size(1)]
    return torch.where(mask.bool(), x, torch.zeros_like(x))


def replace_torch_functions():
    torch.triu = triu_onnx


def restore_torch_functions():
    torch.triu = torch_func["triu"]


def create_onnxruntime_input(vocab_size, batch_size, sequence_length, input_names, data_type=numpy.int64):
    input_ids = numpy.random.randint(low=0, high=vocab_size - 1, size=(batch_size, sequence_length), dtype=data_type)

    inputs = {'input_ids': input_ids}

    if "attention_mask" in input_names:
        attention_mask = numpy.ones([batch_size, sequence_length], dtype=data_type)
        inputs['attention_mask'] = attention_mask

    if "token_type_ids" in input_names:
        segment_ids = numpy.zeros([batch_size, sequence_length], dtype=data_type)
        inputs['token_type_ids'] = segment_ids

    return inputs


def filter_inputs(inputs, input_names):
    remaining_model_inputs = {}
    for input_name in input_names:
        remaining_model_inputs[input_name] = inputs[input_name]
    return remaining_model_inputs


def flatten(inputs):
    return [[flatten(i) for i in inputs] if isinstance(inputs, (list, tuple)) else inputs]


def update_flatten_list(inputs, res_list):
    for i in inputs:
        res_list.append(i) if not isinstance(i, (list, tuple)) else update_flatten_list(i, res_list)
    return res_list


def build_dynamic_axes(example_inputs, outputs_flatten):
    sequence_length = example_inputs["input_ids"].shape[-1]

    dynamic_axes = {key: {0: 'batch_size', 1: 'seq_len'} for key in example_inputs.keys()}

    output_names = ['output_' + str(i + 1) for i in range(len(outputs_flatten))]
    for i, output_name in enumerate(output_names):
        dynamic_axes[output_name] = {0: 'batch_size'}
        dims = outputs_flatten[i].shape
        for j, dim in enumerate(dims):
            if dim == sequence_length:
                dynamic_axes[output_name].update({j: 'seq_len'})
    return dynamic_axes, output_names


def validate_onnx_model(onnx_model_path, example_inputs, example_outputs_flatten, use_gpu, fp16):
    test_session = create_onnxruntime_session(onnx_model_path, use_gpu, enable_all_optimization=False)
    if test_session is None:
        logger.error(f"{onnx_model_path} is an invalid ONNX model")
        return False

    logger.info(f"{onnx_model_path} is a valid ONNX model")

    # Compare the inference result with PyTorch or Tensorflow
    example_ort_inputs = {k: t.cpu().numpy() for k, t in example_inputs.items()}
    example_ort_outputs = test_session.run(None, example_ort_inputs)
    if len(example_outputs_flatten) != len(example_ort_outputs):
        logger.error(
            f"Number of output tensors expected {len(example_outputs_flatten)}, got {len(example_ort_outputs)}")
        return False

    for i in range(len(example_outputs_flatten)):
        abs_diff = numpy.amax(numpy.abs(example_ort_outputs[i] - example_outputs_flatten[i].cpu().numpy()))
        if abs_diff > 1e-4:
            logger.info(f"Max absolute diff={abs_diff} for output tensor {i}")

        rtol = 5e-02 if fp16 else 1e-4
        atol = 1e-01 if fp16 else 1e-4
        if not numpy.allclose(example_ort_outputs[i], example_outputs_flatten[i].cpu(), rtol=rtol, atol=atol):
            logger.error(f"Output tensor {i} is not close: rtol={rtol}, atol={atol}")
            return False

    logger.info(f"inference result of onnxruntime is validated on {onnx_model_path}")
    return True


def get_onnx_file_path(onnx_dir: str, model_name: str, input_count: int, optimized_by_script: bool, use_gpu: bool,
                       precision: Precision, optimized_by_onnxruntime: bool, use_external_data: bool):
    from re import sub
    normalized_model_name = sub(r'[^a-zA-Z0-9_]', '_', model_name)

    if not optimized_by_script:
        filename = f"{normalized_model_name}_{input_count}"
    else:
        device = "gpu" if use_gpu else "cpu"
        filename = f"{normalized_model_name}_{input_count}_{precision}_{device}"

    if optimized_by_onnxruntime:
        filename += f"_ort"

    directory = onnx_dir
    # ONNXRuntime will not write external data so the raw and optimized models shall be in same directory.
    if use_external_data and not optimized_by_onnxruntime:
        directory = os.path.join(onnx_dir, filename)
        if not os.path.exists(directory):
            os.makedirs(directory)

    return os.path.join(directory, f"{filename}.onnx")


def add_filename_suffix(file_path: str, suffix: str) -> str:
    """
    Append a suffix at the filename (before the extension).
    Args:
        path: pathlib.Path The actual path object we would like to add a suffix
        suffix: The suffix to add
    Returns: path with suffix appended at the end of the filename and before extension
    """
    path = Path(file_path)
    return str(path.parent.joinpath(path.stem + suffix).with_suffix(path.suffix))


def optimize_onnx_model_by_ort(onnx_model_path, ort_model_path, use_gpu, overwrite, model_fusion_statistics):
    if overwrite or not os.path.exists(ort_model_path):
        Path(ort_model_path).parent.mkdir(parents=True, exist_ok=True)
        from optimizer import optimize_by_onnxruntime, get_fusion_statistics
        # Use onnxruntime to optimize model, which will be saved to *_ort.onnx
        opt_model = optimize_by_onnxruntime(onnx_model_path,
                                            use_gpu=use_gpu,
                                            optimized_model_path=ort_model_path,
                                            opt_level=99)
        model_fusion_statistics[ort_model_path] = get_fusion_statistics(ort_model_path)
    else:
        logger.info(f"Skip optimization since model existed: {ort_model_path}")


def optimize_onnx_model(onnx_model_path, optimized_model_path, model_type, num_attention_heads, hidden_size, use_gpu,
                        precision, use_raw_attention_mask, overwrite, model_fusion_statistics,
                        use_external_data_format):
    if overwrite or not os.path.exists(optimized_model_path):
        Path(optimized_model_path).parent.mkdir(parents=True, exist_ok=True)

        from optimizer import optimize_model
        from onnx_model_bert import BertOptimizationOptions
        optimization_options = BertOptimizationOptions(model_type)
        optimization_options.use_raw_attention_mask(use_raw_attention_mask)
        if Precision.FLOAT16 == precision:
            optimization_options.enable_gelu_approximation = True
        if Precision.INT8 == precision:
            optimization_options.enable_embed_layer_norm = False

        # Use script to optimize model.
        # Use opt_level <= 1 for models to be converted to fp16, because some fused op (like FusedGemm) has only fp32 and no fp16.
        # It is better to be conservative so we use opt_level=0 here, in case MemcpyFromHost is added to the graph by OnnxRuntime.
        opt_model = optimize_model(onnx_model_path,
                                   model_type,
                                   num_heads=num_attention_heads,
                                   hidden_size=hidden_size,
                                   opt_level=0,
                                   optimization_options=optimization_options,
                                   use_gpu=use_gpu,
                                   only_onnxruntime=False)
        if model_type == 'bert_keras':
            opt_model.use_dynamic_axes()

        model_fusion_statistics[optimized_model_path] = opt_model.get_fused_operator_statistics()

        if Precision.FLOAT16 == precision:
            opt_model.convert_model_float32_to_float16()
        opt_model.save_model_to_file(optimized_model_path, use_external_data_format)
    else:
        logger.info(f"Skip optimization since model existed: {optimized_model_path}")


def modelclass_dispatcher(model_name, custom_model_class):
    if (custom_model_class != None):
        if (custom_model_class in MODEL_CLASSES):
            return custom_model_class
        else:
            raise Exception("Valid model class: " + ' '.join(MODEL_CLASSES))

    if model_name in PRETRAINED_GPT2_MODELS:
        return "GPT2ModelNoPastState"

    import re
    if (re.search('-squad$', model_name) != None):
        return "AutoModelForQuestionAnswering"
    elif (re.search('-mprc$', model_name) != None):
        return "AutoModelForSequenceClassification"
    elif (re.search('gpt2', model_name) != None):
        return "AutoModelWithLMHead"

    return "AutoModel"


def load_pretrained_model(model_name, config, cache_dir, custom_model_class, is_tf_model=False):
    model_class_name = modelclass_dispatcher(model_name, custom_model_class)

    if model_class_name == "GPT2ModelNoPastState":
        if is_tf_model:
            raise NotImplementedError("TFGPT2ModelNoPastState is currently not supported.")
        else:
            return GPT2ModelNoPastState.from_pretrained(model_name, config=config, cache_dir=cache_dir)

    if is_tf_model:
        model_class_name = 'TF' + model_class_name

    transformers_module = __import__("transformers", fromlist=[model_class_name])
    model_class = getattr(transformers_module, model_class_name)

    use_cdn = False if model_name == 't5-11b' else True

    return model_class.from_pretrained(model_name, config=config, cache_dir=cache_dir, use_cdn=use_cdn)


def load_pt_model(model_name, model_class, cache_dir):
    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    if hasattr(config, 'return_dict'):
        config.return_dict = False

    model = load_pretrained_model(model_name, config=config, cache_dir=cache_dir, custom_model_class=model_class)

    return config, model


def load_tf_model(model_name, model_class, cache_dir):
    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)

    model = load_pretrained_model(model_name,
                                  config=config,
                                  cache_dir=cache_dir,
                                  custom_model_class=model_class,
                                  is_tf_model=True)

    return config, model


# For test only
def load_pt_model_from_tf(model_name):
    # Note that we could get pt model from tf, but model source and its structure in this case is different from directly using
    # load_pt_model() and load_tf_model() even with the same name. Therefore it should not be used for comparing with them
    from convert_tf_models_to_pytorch import tf2pt_pipeline
    config, model = tf2pt_pipeline(model_name)

    return config, model
def validate_and_optimize_onnx(model_name, use_external_data_format, model_type, onnx_dir, input_names, use_gpu,
                               precision, optimize_onnx, validate_onnx, use_raw_attention_mask, overwrite, config,
                               model_fusion_statistics, onnx_model_path, example_inputs, example_outputs_flatten):
    is_valid_onnx_model = True
    if validate_onnx:
        is_valid_onnx_model = validate_onnx_model(onnx_model_path, example_inputs, example_outputs_flatten, use_gpu,
                                                  False)

    if optimize_onnx or precision == Precision.FLOAT16 or precision == Precision.INT8:  # Use script (optimizer.py) to optimize
        optimized_model_path = get_onnx_file_path(onnx_dir, model_name, len(input_names), True, use_gpu, precision,
                                                  False, use_external_data_format)
        optimize_onnx_model(onnx_model_path, optimized_model_path, model_type, config.num_attention_heads,
                            config.hidden_size, use_gpu, precision, use_raw_attention_mask, overwrite,
                            model_fusion_statistics, use_external_data_format)

        onnx_model_path = optimized_model_path
        if validate_onnx:
            is_valid_onnx_model = validate_onnx_model(onnx_model_path, example_inputs, example_outputs_flatten, use_gpu,
                                                      precision == Precision.FLOAT16)

        if precision == Precision.INT8:
            logger.info(f"Quantizing model: {onnx_model_path}")
            QuantizeHelper.quantize_onnx_model(onnx_model_path, onnx_model_path, use_external_data_format)
            logger.info(f"Finished quantizing model: {onnx_model_path}")

    else:  # Use OnnxRuntime to optimize
        if is_valid_onnx_model:
            ort_model_path = add_filename_suffix(onnx_model_path, '_ort')
            optimize_onnx_model_by_ort(onnx_model_path, ort_model_path, use_gpu, overwrite, model_fusion_statistics)

    return onnx_model_path, is_valid_onnx_model, config.vocab_size


def export_onnx_model_from_pt(model_name, opset_version, use_external_data_format, model_type, model_class, cache_dir,
                              onnx_dir, input_names, use_gpu, precision, optimize_onnx, validate_onnx,
                              use_raw_attention_mask, overwrite, model_fusion_statistics):

    config, model = load_pt_model(model_name, model_class, cache_dir)
    # config, model = load_pt_model_from_tf(model_name)
    model.cpu()

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    max_input_size = tokenizer.max_model_input_sizes[
        model_name] if model_name in tokenizer.max_model_input_sizes else 1024

    example_inputs = tokenizer.encode_plus("This is a sample input", return_tensors="pt")

    example_inputs = filter_inputs(example_inputs, input_names)

    example_outputs = model(**example_inputs)

    assert isinstance(example_outputs, (list, tuple)), f"type of output is not list or tuple: {type(example_outputs)}"

    # Flatten is needed for gpt2 and distilgpt2.
    example_outputs_flatten = flatten(example_outputs)
    example_outputs_flatten = update_flatten_list(example_outputs_flatten, [])

    onnx_model_path = get_onnx_file_path(onnx_dir, model_name, len(input_names), False, use_gpu, precision, False,
                                         use_external_data_format)

    if overwrite or not os.path.exists(onnx_model_path):
        logger.info("Exporting ONNX model to {}".format(onnx_model_path))
        Path(onnx_model_path).parent.mkdir(parents=True, exist_ok=True)

        dynamic_axes, output_names = build_dynamic_axes(example_inputs, example_outputs_flatten)

        replace_torch_functions()
        torch.onnx.export(model=model,
                          args=tuple(example_inputs.values()),
                          f=onnx_model_path,
                          input_names=list(example_inputs.keys()),
                          output_names=output_names,
                          example_outputs=example_outputs,
                          dynamic_axes=dynamic_axes,
                          do_constant_folding=True,
                          opset_version=opset_version,
                          use_external_data_format=use_external_data_format)
        restore_torch_functions()
    else:
        logger.info(f"Skip export since model existed: {onnx_model_path}")

    onnx_model_file, is_valid_onnx_model, vocab_size = validate_and_optimize_onnx(
        model_name, use_external_data_format, model_type, onnx_dir, input_names, use_gpu, precision, optimize_onnx,
        validate_onnx, use_raw_attention_mask, overwrite, config, model_fusion_statistics, onnx_model_path,
        example_inputs, example_outputs_flatten)

    return onnx_model_file, is_valid_onnx_model, vocab_size, max_input_size


def export_onnx_model_from_tf(model_name, opset_version, use_external_data_format, model_type, model_class, cache_dir,
                              onnx_dir, input_names, use_gpu, precision, optimize_onnx, validate_onnx,
                              use_raw_attention_mask, overwrite, model_fusion_statistics):
    # Use CPU to export
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')

    config, model = load_tf_model(model_name, model_class, cache_dir)

    model._saved_model_inputs_spec = None

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    max_input_size = tokenizer.max_model_input_sizes[
        model_name] if model_name in tokenizer.max_model_input_sizes else 1024

    example_inputs = tokenizer.encode_plus("This is a sample input",
                                           return_tensors="tf",
                                           max_length=max_input_size,
                                           pad_to_max_length=True,
                                           truncation=True)

    example_inputs = filter_inputs(example_inputs, input_names)

    example_outputs = model(example_inputs, training=False)

    # Flatten is needed for gpt2 and distilgpt2.
    example_outputs_flatten = flatten(example_outputs)
    example_outputs_flatten = update_flatten_list(example_outputs_flatten, [])

    onnx_model_path = get_onnx_file_path(onnx_dir, model_name, len(input_names), False, use_gpu, precision, False,
                                         use_external_data_format)

    if overwrite or not os.path.exists(onnx_model_path):
        logger.info("Exporting ONNX model to {}".format(onnx_model_path))
        Path(onnx_model_path).parent.mkdir(parents=True, exist_ok=True)

        import keras2onnx
        onnx_model = keras2onnx.convert_keras(model, model.name, target_opset=opset_version)
        keras2onnx.save_model(onnx_model, onnx_model_path)
    else:
        logger.info(f"Skip export since model existed: {onnx_model_path}")

    model_type = model_type + '_keras'

    onnx_model_file, is_valid_onnx_model, vocab_size = validate_and_optimize_onnx(
        model_name, use_external_data_format, model_type, onnx_dir, input_names, use_gpu, precision, optimize_onnx,
        validate_onnx, use_raw_attention_mask, overwrite, config, model_fusion_statistics, onnx_model_path,
        example_inputs, example_outputs_flatten)

    return onnx_model_file, is_valid_onnx_model, vocab_size, max_input_size

