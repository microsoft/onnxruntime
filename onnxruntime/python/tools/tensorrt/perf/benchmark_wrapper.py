import os
import csv
import logging
import coloredlogs
import argparse
import copy
import json
import re
import sys
import pprint
from benchmark import *

def write_model_info_to_file(model, path):
    with open(path, 'w') as file:
        file.write(json.dumps(model)) # use `json.loads` to do the reverse

def main():
    args = parse_arguments()
    setup_logger(False)
    pp = pprint.PrettyPrinter(indent=4)

    models = {}
    if ".json" in args.model_source:
        # logger.info("Parsing model information from json file ...\n")
        parse_models_info_from_file(args.model_source, models)
    else:
        # logger.info("Parsing model information from specified directory ...\n")
        parse_models_info_from_directory(args.model_source, models)

    fail_results = []
    model_ep_fail_map = {}

    benchmark_fail_csv = 'benchmark_fail.csv' 
    benchmark_ratio_csv = 'benchmark_ratio.csv'
    benchmark_success_csv = 'benchmark_success.csv' 
    benchmark_latency_csv = 'benchmark_latency.csv'

    for model, model_info in models.items():
        logger.info("\n\n================= " + model +" =====================")

        model_info["model_name"] = model 
        
        tmp_directory = os.path.join(os.getcwd(), "tmp")
        if not os.path.exists(tmp_directory):
            os.mkdir(tmp_directory) 

        model_list_file = os.path.join(tmp_directory, model +'.json')
        write_model_info_to_file([model_info], model_list_file)


        ep_list = ["CUDAExecutionProvider", "TensorrtExecutionProvider", "CUDAExecutionProvider_fp16", "TensorrtExecutionProvider_fp16"]

        for ep in ep_list:
            if args.running_mode == "validate":
                p = subprocess.run(["python3",
                                    "benchmark.py",
                                    "-r", args.running_mode,
                                    "-m", model_list_file,
                                    "--ep", ep,
                                    "-s", args.symbolic_shape_infer,
                                    "-o", args.perf_result_path,
                                    "--benchmark_fail_csv", benchmark_fail_csv,
                                    "--benchmark_ratio_csv", benchmark_ratio_csv])
            elif args.running_mode == "benchmark":
                p = subprocess.run(["python3",
                                    "benchmark.py",
                                    "-r", args.running_mode,
                                    "-m", model_list_file,
                                    "--ep", ep,
                                    "-s", args.symbolic_shape_infer,
                                    "-t", str(args.test_times),
                                    "-o", args.perf_result_path,
                                    "--need_write_result", "false",
                                    "--benchmark_latency_csv", benchmark_latency_csv,
                                    "--benchmark_success_csv", benchmark_success_csv])
            logger.info(p)

            if p.returncode != 0:
                error_type = "runtime error" 
                error_message = "perf script exited with returncode = " + str(p.returncode)
                logger.error(error_message)

                update_fail_model(model_ep_fail_map, fail_results, model, ep, error_type, error_message)

    path = os.path.join(os.getcwd(), args.perf_result_path)
    if not os.path.exists(path):
        from pathlib import Path
        Path(path).mkdir(parents=True, exist_ok=True)

    if args.running_mode == "validate":
        if len(model_ep_fail_map) > 0:
            write_model_ep_fail_map_to_file(model_ep_fail_map)

        if fail_results:
            csv_filename = os.path.join(path, benchmark_fail_csv)
            output_fail(fail_results, csv_filename)
    elif args.running_mode == "benchmark":
        latency_comparison_map = read_map_from_file(".benchmark_latency_map")
        add_improvement_information(latency_comparison_map)
        output_latency(latency_comparison_map, os.path.join(path, benchmark_latency_csv))

    get_system_info()

if __name__ == "__main__":
    main()
