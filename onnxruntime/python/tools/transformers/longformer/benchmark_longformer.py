# Please run convert_longformer_to_onnx.py to get onnx model before running this script.
# Tested with python 3.7, onnxruntime-gpu 1.6.0 (or nightly), PyTorch 1.7.0, transformers 4.0, CUDA 10.2, CUDNN 8.0
# Example step by step command lines for benchmarking longformer base model (without/with optimizer) in Linux:
#   python setup.py install
#   python convert_longformer_to_onnx.py -m longformer-base-4096
#   python benchmark_longformer.py -m longformer-base-4096
#   python convert_longformer_to_onnx.py -m longformer-base-4096 -o
#   python benchmark_longformer.py -m longformer-base-4096

import timeit
from datetime import datetime
import csv
import argparse
import os
import sys
import torch
import onnxruntime

# Mapping from model name to pretrained model name
MODELS = {
    "longformer-base-4096": "allenai/longformer-base-4096",
    "longformer-random-tiny": "patrickvonplaten/longformer-random-tiny"  # A tiny model for debugging
}

is_debug = False

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Run onnx model with ORT
import benchmark_helper

def get_dummy_inputs(sequence_length, num_global_tokens, device):
    # Create dummy inputs
    input_ids = torch.arange(sequence_length).unsqueeze(0).to(device)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long,
                                device=input_ids.device)  # TODO: use random word ID. #TODO: simulate masked word
    global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long, device=input_ids.device)
    global_token_index = list(range(num_global_tokens))
    global_attention_mask[:, global_token_index] = 1
    # TODO: support more inputs like token_type_ids, position_ids
    return input_ids, attention_mask, global_attention_mask


def diff_outputs(ort_outputs, torch_outputs):
    max_diff = []
    # Compare the outputs to find max difference
    for i in range(2):
        print(f"output {i} shape: ORT={ort_outputs[i].shape}, Torch={torch_outputs[i].shape}")
        diff = (torch.from_numpy(ort_outputs[i]) - torch_outputs[i].to('cpu')).abs().max()
        max_diff.append(diff)
    print(f"max diff for output: {max_diff}")
    return max_diff


def test_torch(device, model, model_name, batch_sizes, sequence_lengths, global_lengths, test_times, num_threads):
    # Comment the following so that PyTorch use default setting as well.
    #if num_threads <= 0:
    #    import psutil
    #    num_threads = psutil.cpu_count(logical=False)
    if num_threads > 0:
        torch.set_num_threads(num_threads)

    results = []
    for batch_size in batch_sizes:
        for sequence_length in sequence_lengths:  # This is total length of <query, document>.
            for global_length in global_lengths:  # This is length of <query>. Short query (8) for search keywords, and longer query (16) for question like
                print(f"batch_size={batch_size} sequence_length={sequence_length} global_length={global_length}...")
                input_ids, attention_mask, global_attention_mask = get_dummy_inputs(sequence_length, global_length,
                                                                                    device)

                # Run PyTorch
                _ = model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
                runtimes = timeit.repeat(lambda: model(input_ids, attention_mask, global_attention_mask),
                                         repeat=test_times,
                                         number=1)
                result = {
                    "engine": "torch",  #TODO: test torchscript
                    "version": torch.__version__,
                    "device": "cuda",
                    "optimizer": "",
                    "precision": "fp32",
                    "io_binding": "",
                    "model_name": model_name,
                    "inputs": 3,
                    "threads": num_threads,
                    "batch_size": batch_size,
                    "sequence_length": sequence_length,
                    "global_length": global_length,
                    "datetime": str(datetime.now()),
                }
                result.update(benchmark_helper.get_latency_result(runtimes, batch_size))

                print(result)
                results.append(result)
    return results


def test_onnxruntime(device,
                     model,
                     model_name,
                     ort_session,
                     batch_sizes,
                     sequence_lengths,
                     global_lengths,
                     test_times,
                     num_threads,
                     optimizer=False,
                     precision='fp32'):
    results = []
    for batch_size in batch_sizes:
        for sequence_length in sequence_lengths:  # This is total length of <query, document>.
            for global_length in global_lengths:  # This is length of <query>. Short query (8) for search keywords, and longer query (16) for question like
                print(
                    f"Testing batch_size={batch_size} sequence_length={sequence_length} global_length={global_length} optimizer={optimizer}, precision={precision}..."
                )
                input_ids, attention_mask, global_attention_mask = get_dummy_inputs(sequence_length, global_length,
                                                                                    device)

                # Run OnnxRuntime
                ort_inputs = {
                    "input_ids": input_ids.cpu().numpy(),
                    "attention_mask": attention_mask.cpu().numpy(),
                    "global_attention_mask": global_attention_mask.cpu().numpy()
                }

                # run one query for warm up
                ort_outputs = ort_session.run(None, ort_inputs)

                if is_debug:
                    # Run PyTorch then compare the results with OnnxRuntime.
                    torch_outputs = model(input_ids,
                                          attention_mask=attention_mask,
                                          global_attention_mask=global_attention_mask)

                    max_diff = diff_outputs(ort_outputs, torch_outputs)
                    print("max diff for outputs", max_diff)
                    if max(max_diff) > 0.001:
                        print("ort_inputs", ort_inputs)
                        print("ort_outputs", ort_outputs)

                device = input_ids.device
                result_template = {
                    "model_name": model_name,
                    "inputs": 3,
                    "engine": "OnnxRuntime",
                    "version": onnxruntime.__version__,
                    "device": "cuda",
                    "precision": precision,
                    "optimizer": optimizer,
                    "threads": num_threads,
                    "batch_size": batch_size,
                    "sequence_length": sequence_length,
                    "global_length": global_length,
                    "test_times": test_times,
                    "datetime": str(datetime.now()),
                }

                max_last_state_size = max(batch_sizes) * max(sequence_lengths) * model.config.hidden_size
                max_pooler_size = max(batch_sizes) * max(sequence_lengths)

                result = benchmark_helper.inference_ort_with_io_binding(
                    ort_session,
                    ort_inputs,
                    result_template=result_template,
                    repeat_times=test_times,
                    ort_output_names=["last_state", "pooler"],
                    ort_outputs=ort_outputs,
                    output_buffers=[],
                    output_buffer_max_sizes=[max_last_state_size, max_pooler_size],
                    batch_size=batch_size,
                    device=device)
                print(result)
                results.append(result)
    return results


def test_all(args):
    # Currently, the longformer attention operator could only run in GPU (no CPU implementation yet).
    device = torch.device('cuda:0')

    results = []
    for model_name in args.models:
        # Here we run an example input
        from transformers import LongformerModel
        torch_model_name_or_dir = MODELS[model_name]
        model = LongformerModel.from_pretrained(torch_model_name_or_dir)  # pretrained model name or directory
        model.to(device)

        # Search onnx model in the following order: optimized fp16 model, optimized fp32 model, raw model
        optimized = False
        precision = 'fp32'
        onnx_model_path = model_name + ".onnx"
        optimized_fp32_model = model_name + "_fp32.onnx"
        optimized_fp16_model = model_name + "_fp16.onnx"
        import os.path
        if os.path.isfile(optimized_fp16_model):
            onnx_model_path = optimized_fp16_model
            optimized = True
            precision = 'fp16'
        elif os.path.isfile(optimized_fp32_model):
            onnx_model_path = optimized_fp32_model
            optimized = True

        for num_threads in args.num_threads:
            if "torch" in args.engines:
                results += test_torch(device, model, model_name, args.batch_sizes, args.sequence_lengths,
                                      args.global_lengths, args.test_times, num_threads)

            if "onnxruntime" in args.engines:
                session = benchmark_helper.create_onnxruntime_session(onnx_model_path,
                                                                      use_gpu=True,
                                                                      enable_all_optimization=True,
                                                                      num_threads=num_threads)
                results += test_onnxruntime(device, model, model_name, session, args.batch_sizes, args.sequence_lengths,
                                            args.global_lengths, args.test_times, num_threads, optimized, precision)
    return results


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-m",
                        "--models",
                        required=False,
                        nargs="+",
                        type=str,
                        default=["longformer-random-tiny"] if is_debug else ["longformer-base-4096"],
                        choices=list(MODELS.keys()),
                        help="Pre-trained models in the list: " + ", ".join(MODELS.keys()))

    parser.add_argument("-e",
                        "--engines",
                        required=False,
                        nargs="+",
                        type=str,
                        default=['onnxruntime'],
                        choices=['onnxruntime', 'torch'],
                        help="Engines to benchmark. For large model, recommend to test only one engine at a time.")

    parser.add_argument("-t",
                        "--test_times",
                        required=False,
                        default=1000,
                        type=int,
                        help="Number of repeat times to get average inference latency.")

    parser.add_argument("-b", "--batch_sizes", nargs="+", type=int, default=[1])

    # If multiple of window size is used during exporting onnx model, there is no padding in ONNX model so you will need padding by yourself before running onnx model.
    # In that case, you can only test sequence length that is multiple of window size (4 or 512 for these two models).
    parser.add_argument("-s",
                        "--sequence_lengths",
                        nargs="+",
                        type=int,
                        default=[4] if is_debug else [512, 1024, 2048, 4096])

    parser.add_argument("-g", "--global_lengths", nargs="+", type=int, default=[1] if is_debug else [8])

    parser.add_argument("-n", "--num_threads", required=False, nargs="+", type=int, default=[0], help="Threads to use")

    args = parser.parse_args()
    return args


def output_summary(results, csv_filename, args):
    with open(csv_filename, mode="a", newline='') as csv_file:
        header_names = [
            "model_name", "inputs", "engine", "version", "device", "precision", "optimizer", "io_binding", "threads"
        ]
        data_names = []
        for batch_size in args.batch_sizes:
            for sequence_length in args.sequence_lengths:
                for global_length in args.global_lengths:
                    data_names.append(f"b{batch_size}_s{sequence_length}_g{global_length}")

        csv_writer = csv.DictWriter(csv_file, fieldnames=header_names + data_names)
        csv_writer.writeheader()
        for model in args.models:
            for input_count in [1, 2, 3]:
                for engine_name in args.engines:
                    for io_binding in [True, False, ""]:
                        for threads in args.num_threads:
                            row = {}
                            for result in results:
                                if result["model_name"] == model and result["inputs"] == input_count and result[
                                        "engine"] == engine_name and result["io_binding"] == io_binding and result[
                                            "threads"] == threads:
                                    headers = {k: v for k, v in result.items() if k in header_names}
                                    if not row:
                                        row.update(headers)
                                        row.update({k: "" for k in data_names})
                                    else:
                                        for k in header_names:
                                            assert row[k] == headers[k]
                                    b = result["batch_size"]
                                    s = result["sequence_length"]
                                    g = result["global_length"]
                                    row[f"b{b}_s{s}_g{g}"] = result["average_latency_ms"]
                            if row:
                                csv_writer.writerow(row)

    print(f"Summary results are saved to csv file: {csv_filename}")


def output_details(results, csv_filename):
    with open(csv_filename, mode="a", newline='') as csv_file:
        column_names = [
            "engine", "version", "device", "precision", "optimizer", "io_binding", "model_name", "inputs", "threads",
            "batch_size", "sequence_length", "global_length", "datetime", "test_times", "QPS", "average_latency_ms",
            "latency_variance", "latency_90_percentile", "latency_95_percentile", "latency_99_percentile"
        ]

        csv_writer = csv.DictWriter(csv_file, fieldnames=column_names)
        csv_writer.writeheader()
        for result in results:
            csv_writer.writerow(result)

    print(f"Detail results are saved to csv file: {csv_filename}")


def main():
    args = parse_arguments()

    assert len(args.models) == 1, "run only one model at a time"

    if not torch.cuda.is_available():
        raise RuntimeError("Please install PyTorch with Cuda, and use a machine with GPU for testing gpu performance.")

    torch.set_grad_enabled(False)

    all_results = test_all(args)

    time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_filename = f"benchmark_detail_{time_stamp}.csv"
    output_details(all_results, csv_filename)

    csv_filename = f"benchmark_summary_{time_stamp}.csv"
    output_summary(all_results, csv_filename, args)


if __name__ == "__main__":
    main()
