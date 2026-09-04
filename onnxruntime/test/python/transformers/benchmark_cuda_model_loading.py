# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Measure CUDA InferenceSession creation for a model with external data.

Example:
  python benchmark_cuda_model_loading.py \
      --model /path/to/model.onnx \
      --external-data /path/to/model.onnx.data \
      --evict-file-cache

Run this script in a fresh process for every sample. Cache eviction is applied
only to the paths passed on the command line and remains an operating-system
hint rather than a guarantee.
"""

import argparse
import json
import os
import time

import onnxruntime


def reading_thread_count(value):
    count = int(value)
    if not 1 <= count <= 64:
        raise argparse.ArgumentTypeError("must be between 1 and 64")
    return count


def evict_file_cache(path):
    if not hasattr(os, "posix_fadvise"):
        raise RuntimeError("file-cache eviction requires os.posix_fadvise")

    with open(path, "rb") as file:
        os.posix_fadvise(file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def main():
    parser = argparse.ArgumentParser(description="Benchmark CUDA model loading")
    parser.add_argument("--model", required=True, help="Path to the ONNX model")
    parser.add_argument(
        "--external-data",
        action="append",
        default=[],
        help="External-data file to evict with --evict-file-cache; may be specified more than once",
    )
    parser.add_argument("--device-id", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--threads", type=int, default=96, help="Intra-op thread count")
    parser.add_argument(
        "--reading-threads",
        type=reading_thread_count,
        help="Override parallel reads per CUDA pinned staging buffer (runtime default: 4)",
    )
    parser.add_argument(
        "--evict-file-cache",
        action="store_true",
        help="Advise the OS to evict the model and external-data files before creating the session",
    )
    args = parser.parse_args()

    paths = [args.model, *args.external_data]
    for path in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    if args.evict_file_cache:
        for path in paths:
            evict_file_cache(path)

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    if args.reading_threads is not None:
        options.add_session_config_entry(
            "session.cuda.external_data_loader_reading_threads",
            str(args.reading_threads),
        )

    start = time.perf_counter()
    session = onnxruntime.InferenceSession(
        args.model,
        sess_options=options,
        providers=[
            ("CUDAExecutionProvider", {"device_id": args.device_id}),
            "CPUExecutionProvider",
        ],
    )
    elapsed = time.perf_counter() - start

    print(
        json.dumps(
            {
                "active_providers": session.get_providers(),
                "device_id": args.device_id,
                "evict_file_cache": args.evict_file_cache,
                "model": os.path.abspath(args.model),
                "reading_threads": args.reading_threads or 4,
                "seconds": elapsed,
                "threads": args.threads,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
