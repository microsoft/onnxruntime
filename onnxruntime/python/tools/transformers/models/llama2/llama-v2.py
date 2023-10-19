import torch
import os
import psutil
import time
import argparse
import onnx

import onnxruntime as ort

from collections import OrderedDict
from functools import partial
from onnxruntime.transformers.onnx_model import OnnxModel
from transformers import LlamaConfig, LlamaTokenizer, LlamaForCausalLM
from optimum.onnx import merge_decoders as optimum_merge_decoders

from export_to_onnx import run_torchscript_export, optimize_onnx_model
from ort_llama import OrtModelForLlamaCausalLM
from dist_settings import init_dist, get_rank, get_size, barrier, print_out

init_dist()
from modeling import patching_llama  # noqa: E402

import torch.nn.init

import logging

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

torch.nn.init.kaiming_uniform_ = lambda x, *args, **kwargs: x
torch.nn.init.uniform_ = lambda x, *args, **kwargs: x
torch.nn.init.kaiming_normal_ = lambda x, *args, **kwargs: x

DECODER_MODEL = "rank-{}_decoder_model_fp32.onnx"
DECODER_PAST_MODEL = "rank-{}_decoder_with_past_model_fp32.onnx"
MERGED_MODEL = "rank-{}_decoder_merged_model_fp32.onnx"


def setup_session_option(args, local_rank):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = psutil.cpu_count(logical=False)
    so.log_severity_level = 4  # 0 for verbose
    if args.verbose:
        so.log_severity_level = 0
        ort.set_default_logger_severity(0)  # open log
        ort.set_default_logger_verbosity(1000)  # verbose

    if args.save_opt:
        so.optimized_model_filepath = f"ort-opted-rank-{local_rank}-{args.output_name}.onnx"

    if args.profile and local_rank == 0:
        so.enable_profiling = args.profile
        so.profile_file_prefix = f"ort-profile-rank-{local_rank}"

    provider_opt = {"device_id": local_rank, "tunable_op_enable": args.tunable, "tunable_op_tuning_enable": args.tuning}

    return so, provider_opt


def setup_ort_model(args, rank):
    config = LlamaConfig.from_pretrained(args.model)
    if args.layer2:
        config.num_hidden_layers = 2
    if args.merge:
        model_f = MERGED_MODEL.format(rank)
    else:
        decoder_model = DECODER_MODEL.format(rank)
        decoder_past_model = DECODER_PAST_MODEL.format(rank)
        model_f = (decoder_model, decoder_past_model)
    sess_opt, provider_opt = setup_session_option(args, rank)

    model = OrtModelForLlamaCausalLM(
        args,
        model_f,
        rank,
        sess_opt,
        provider_opt,
        config=config,
    )
    model.to(torch.device(rank))

    return model


def setup_torch_model(args, use_cuda=True):
    barrier()
    world_size = get_size()
    rank = get_rank()
    for i in range(world_size):
        if i == rank:
            config = LlamaConfig.from_pretrained(args.model)
            if args.layer2:
                config.num_hidden_layers = 2
            model = LlamaForCausalLM.from_pretrained(args.model, torch_dtype=config.torch_dtype, config=config)
            if world_size > 1:
                model.parallel_model()
            if use_cuda:
                model.to(torch.device(rank))
            model.eval()
            model.requires_grad_(False)
            if args.cugraph:
                model.addition_init(1, model.device, config.torch_dtype, config.max_position_embeddings)
            if args.compile:
                if args.custom_gen:
                    model = torch.compile(model)
                else:
                    model.generate = torch.compile(model.generate, backend="inductor")
        barrier()
    return model


def export_model(args):
    rank = get_rank()
    config = LlamaConfig.from_pretrained(args.model)
    if args.layer2:
        config.num_hidden_layers = 2
    world_size = get_size()

    model = setup_torch_model(args, use_cuda=True)
    barrier()
    for i in range(world_size):
        if i == rank:
            decoder_model_fp32, decoder_with_past_fp32 = run_torchscript_export(args, config, model, rank, world_size)
            # convert to fp16
            if args.convert_fp16:
                decoder_model_fp16_path = f"rank-{rank}_decoder_model_fp16.onnx"
                model = OnnxModel(onnx.load_model(decoder_model_fp32, load_external_data=True))
                model.convert_float_to_float16(keep_io_types=False, op_block_list=["If"])
                model.save_model_to_file(
                    decoder_model_fp16_path, use_external_data_format=True, all_tensors_to_one_file=True
                )
                del model

                # Convert decoder_with_past_model.onnx to FP16
                decoder_with_past_model_fp16_path = f"rank-{rank}_decoder_with_past_model_fp16.onnx"
                model = OnnxModel(onnx.load_model(decoder_with_past_fp32, load_external_data=True))
                model.convert_float_to_float16(keep_io_types=False, op_block_list=["If"])
                model.save_model_to_file(
                    decoder_with_past_model_fp16_path,
                    use_external_data_format=True,
                    all_tensors_to_one_file=True,
                )
                del model

        barrier()


def custom_generate(model, input_ids, attention_mask, max_new_tokens=128, **kwargs):
    past_kvs = None
    output_ids = [input_ids]
    tokens = input_ids
    for _ in range(max_new_tokens):
        model_inputs = model.prepare_inputs_for_generation(
            tokens, attention_mask=attention_mask, past_key_values=past_kvs, **kwargs
        )
        # model_inputs.update({"output_attentions": False, "output_hidden_states": False})
        results = model(**model_inputs)
        logits = results.logits
        past_kvs = results.past_key_values

        attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)

        # greedy search
        tokens = torch.argmax(logits[:, -1, :], dim=1, keepdim=True)

        output_ids.append(tokens)
    return torch.concat(output_ids, dim=1)


decoder_generate_graph = None
graph_inputs = None
graph_outputs = None


def torch_generate_with_graph(args, model, input_ids, attention_mask, max_new_tokens=128, use_cache=True, **kwargs):
    past_kvs = None
    output_ids = [input_ids]
    tokens = input_ids
    _, pos = input_ids.shape
    position_ids = None
    for i in range(max_new_tokens):
        model_inputs = model.prepare_inputs(
            tokens,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_kvs,
            use_cache=use_cache,
            **kwargs,
        )

        torch.cuda.nvtx.range_push("forward")
        if past_kvs is not None and args.cugraph:
            # in decoding phase
            global decoder_generate_graph
            global graph_inputs
            global graph_outputs
            if decoder_generate_graph is None:
                # copy input_ids and position into another memory to be re-used by graph
                model_inputs["input_ids"] = torch.empty_like(model_inputs["input_ids"]).copy_(model_inputs["input_ids"])
                model_inputs["position_ids"] = torch.empty_like(model_inputs["position_ids"]).copy_(
                    model_inputs["position_ids"]
                )

                # warm up for graph
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    model(**model_inputs)
                torch.cuda.current_stream().wait_stream(s)

                # record graph
                g1 = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g1):
                    results = model(**model_inputs)
                graph_inputs = model_inputs
                graph_outputs = results
                decoder_generate_graph = g1
            # copy inputs into graph_inputs
            # only input_ids and position need to be copied.
            # attention_mask is modified by prepare_inputs
            # past_kv is modified by attention layer
            for k in ["input_ids", "position_ids"]:
                graph_inputs[k].copy_(model_inputs[k])

            decoder_generate_graph.replay()
            results = graph_outputs
        else:
            results = model(**model_inputs)

        torch.cuda.nvtx.range_pop()

        logits = results.logits
        past_kvs = results.past_key_values
        position_ids = torch.tensor([pos], dtype=torch.long, device=model.device)
        position_ids.unsqueeze(0)
        pos += 1

        # greedy search
        tokens = torch.argmax(logits[:, -1, :], dim=1, keepdim=True)

        output_ids.append(tokens)
    return torch.concat(output_ids, dim=1)


def _run_gen(args, model, tokenizer, input_ids, attention_mask, name):
    if args.custom_gen:
        print_out("[generate] Using custom_generate")
        if args.cugraph and args.torch:
            gen = partial(torch_generate_with_graph, args, model)
        else:
            gen = partial(custom_generate, model)
    else:
        print_out("[generate] Using original model.generate")
        gen = model.generate

    outputs = gen(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=128,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,  # Default value is True in latest config
        temperature=1.0,
        top_p=1.0,
        use_cache=True,
    )

    print_out("input ids size: ", input_ids.shape, " value: ", input_ids)
    print_out("output size: ", outputs[0].shape, " value: ", outputs[0])
    response = tokenizer.decode(outputs[0][1:], skip_special_token=True)
    print_out(f"[{name}] Response:", response)


def run_generate(args, local_rank):
    tokenizer = LlamaTokenizer.from_pretrained(args.model)
    # prompt = "Q: What is the largest animal?\nA:"
    # prompt = "Once upon a time,"
    prompt = "Q: there are two sets of nodes in a graph, each node in one set is connecting to all nodes in the other set, what is graph called?\nA:"
    inputs = tokenizer(prompt, return_tensors="pt")
    if args.ort:
        ort_model = setup_ort_model(args, local_rank)
        input_ids = inputs.input_ids.to(ort_model.device)
        attention_mask = inputs.attention_mask.to(ort_model.device)
        _run_gen(args, ort_model, tokenizer, input_ids, attention_mask, "ORT")

    if args.torch:
        torch_model = setup_torch_model(args, use_cuda=True)
        input_ids = inputs.input_ids.to(torch_model.device)
        attention_mask = inputs.attention_mask.to(torch_model.device)
        _run_gen(args, torch_model, tokenizer, input_ids, attention_mask, "Torch")


def func_benchmark(fn, warm=5, steps=10):
    for _ in range(warm):
        torch.cuda.nvtx.range_push("gen warmup")
        fn()
        torch.cuda.nvtx.range_pop()

    start = time.time()
    for i in range(steps):
        torch.cuda.nvtx.range_push(f"gen step{i}")
        fn()
        torch.cuda.nvtx.range_pop()
    cost = time.time() - start
    return cost / steps


def _print_stat(p_len, cost, gen_lens):
    prompt_cost = cost[0]
    token_cost = (cost[1] - cost[0]) / (gen_lens[1] - gen_lens[0])
    print_out(f"prompt_len: {p_len:4}, prompt_cost: {prompt_cost:.4f}, token_cost: {token_cost:.4f}")


def _print_stats(name, costs, gen_lens):
    print_out(f"{name} ==============================================")
    for k, v in costs.items():
        _print_stat(k, v, gen_lens)


def _run_bmk(args, model, name):
    tokenizer = LlamaTokenizer.from_pretrained(args.model)
    # print(tokenizer.pad_token_id, tokenizer.eos_token_id)

    batch = 1
    prompt_lens = [32, 64, 128, 256, 512, 1024, 2048]
    # prompt_lens = [32]
    generate_lens = [1, 33]

    if args.custom_gen:
        print_out("[benchmark] Using custom_generate")
        if args.cugraph and args.torch:
            gen = partial(torch_generate_with_graph, args, model)
        else:
            gen = partial(custom_generate, model)
    else:
        print_out("[benchmark] Using original model.generate")
        gen = model.generate

    if args.cugraph:
        print_out("[benchmark] CUDAGraph enabled")

    costs = OrderedDict()
    for p_len in prompt_lens:
        cost = []
        for gen_len in generate_lens:
            # generate input prompt
            prompt = " ".join(["Hello"] * p_len)

            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, padding=False).to(model.device)
            gen_func = lambda: gen(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=gen_len,
                min_new_tokens=gen_len,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,  # Default value is True in latest config
                temperature=1.0,
                top_p=1.0,
                use_cache=True,
            )

            item = func_benchmark(gen_func, warm=args.warm, steps=args.loop_cnt)
            cost.append(item)
        costs[p_len] = cost
        if len(generate_lens) > 1:
            _print_stat(p_len, cost, generate_lens)
    if len(generate_lens) > 1:
        _print_stats(name, costs, generate_lens)


def run_benchmark(args, local_rank):
    if args.ort:
        ort_model = setup_ort_model(args, local_rank)
        _run_bmk(args, ort_model, "ORT")

    if args.torch:
        torch_model = setup_torch_model(args, use_cuda=True)
        _run_bmk(args, torch_model, "Torch")


def main(args):
    local_rank = get_rank()
    torch.cuda.set_device(local_rank)
    decoder_model = DECODER_MODEL.format(local_rank)
    decoder_past_model = DECODER_PAST_MODEL.format(local_rank)
    merged_model = MERGED_MODEL.format(local_rank)

    if args.export:
        export_model(args)
        print("[export] Completed.")

    if args.optimize:
        if not (args.merge and os.path.exists(merged_model)):
            optimize_onnx_model(local_rank)

    if args.merge:
        if not os.path.exists(merged_model):
            print("merging decoder models")
            optimum_merge_decoders(decoder_model, decoder_past_model, save_path=merged_model)
            print("merging decoder models completed")

    if args.generate:
        run_generate(args, local_rank)

    if args.benchmark:
        run_benchmark(args, local_rank)


def get_args():
    parser = argparse.ArgumentParser(description="Example for Fine-tuning with PyTorch Templates")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf", help="pretrained model name")
    parser.add_argument("--output-name", type=str, default=".", help="name of the output directory")
    parser.add_argument("-l", "--loop-cnt", type=int, default=10, help="number of timing steps during benchmarking")
    parser.add_argument("-w", "--warm", type=int, default=5, help="number of warmup steps during benchmarking")
    parser.add_argument("-o", "--ort", action="store_true", help="run using ONNX Runtime")
    parser.add_argument("-t", "--torch", action="store_true", help="run using PyTorch")
    parser.add_argument("-g", "--generate", action="store_true", help="run model generate")
    parser.add_argument("-b", "--benchmark", action="store_true", help="run model benchmarking")
    parser.add_argument("--batch", type=int, default=1, help="batch size")
    parser.add_argument("--custom-gen", action="store_true", help="enable custom generate function")
    parser.add_argument("--layer2", action="store_true", help="set num_hidden_layers to 2 (for debugging/testing)")

    ort_group = parser.add_argument_group("ORT specific arguments")

    ort_group.add_argument("-e", "--export", action="store_true", help="export the model")
    ort_group.add_argument("--optimize", action="store_true", help="enable graph optimization")
    ort_group.add_argument("--convert-fp16", action="store_true", help="convert model to float16 (fp16)")
    ort_group.add_argument("--save-opt", action="store_true", help="save optimized ONNX model")
    ort_group.add_argument("--merge", action="store_true", help="merge decoders into one model")
    ort_group.add_argument("--profile", action="store_true", help="enable profiling")
    ort_group.add_argument("--tunable", action="store_true", help="enable TunableOp")
    ort_group.add_argument("--tuning", action="store_true", help="enable tuning for TunableOp")
    ort_group.add_argument("--provider", type=str, default="rocm", help="specify execution provider")
    ort_group.add_argument("-v", "--verbose", action="store_true", help="enable verbose logging")

    torch_group = parser.add_argument_group("PyTorch specific arguments")

    torch_group.add_argument("--compile", action="store_true", help="enable torch.compile")
    torch_group.add_argument("--cugraph", action="store_true", help="enable CUDAGraph")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.output_name, exist_ok=True)
    os.chdir(args.output_name)

    if args.optimize:
        # args.export = True
        DECODER_MODEL = f"opt_{DECODER_MODEL}"
        DECODER_PAST_MODEL = f"opt_{DECODER_PAST_MODEL}"
        MERGED_MODEL = f"opt_{MERGED_MODEL}"

    if args.cugraph and not args.custom_gen:
        print_out("WARNING: CUDAGraph enabled but custom_gen is not, setting it to True")
        args.custom_gen = True

    main(args)
