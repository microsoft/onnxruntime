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
from perf_utils import get_latest_commit_hash

def write_model_info_to_file(model, path):
    with open(path, 'w') as file:
        file.write(json.dumps(model)) # use `json.loads` to do the reverse

def main():
    args = parse_arguments()
    setup_logger(False)
    pp = pprint.PrettyPrinter(indent=4)

    models = {}
    if ".json" in args.model_source:
        parse_models_info_from_file(args.model_source, models)
    else:
        parse_models_info_from_directory(args.model_source, models)

    model_to_fail_ep = {}

    commit = get_latest_commit_hash()
    benchmark_fail_csv = 'fail_' + commit + '.csv'  
    benchmark_metrics_csv = 'metrics_' + commit + '.csv'
    benchmark_success_csv = 'success_' + commit + '.csv' 
    benchmark_latency_csv = 'latency_' + commit + '.csv'

    for model, model_info in models.items():
        logger.info("\n" + "="*40 + "="*len(model))
        logger.info("="*20 + model +"="*20)
        logger.info("="*40 + "="*len(model))

        model_info["model_name"] = model 
        
        model_list_file = os.path.join(os.getcwd(), model +'.json')
        write_model_info_to_file([model_info], model_list_file)


        ep_list = ["CPUExecutionProvider", "CUDAExecutionProvider", "TensorrtExecutionProvider", "CUDAExecutionProvider_fp16", "TensorrtExecutionProvider_fp16"]

        for ep in ep_list:
            if args.running_mode == "validate":
                p = subprocess.run(["python3",
                                    "benchmark.py",
                                    "-r", args.running_mode,
                                    "-m", model_list_file,
                                    "--ep", ep,
                                    "-o", args.perf_result_path,
                                    "--write_test_result", "false",
                                    "--benchmark_fail_csv", benchmark_fail_csv,
                                    "--benchmark_metrics_csv", benchmark_metrics_csv])
            elif args.running_mode == "benchmark":
                p = subprocess.run(["python3",
                                    "benchmark.py",
                                    "-r", args.running_mode,
                                    "-m", model_list_file,
                                    "--ep", ep,
                                    "-t", str(args.test_times),
                                    "-o", args.perf_result_path,
                                    "--write_test_result", "false",
                                    "--benchmark_latency_csv", benchmark_latency_csv,
                                    "--benchmark_success_csv", benchmark_success_csv])
            logger.info(p)

            if p.returncode != 0:
                error_type = "runtime error" 
                error_message = "perf script exited with returncode = " + str(p.returncode)
                logger.error(error_message)

                update_fail_model_map(model_to_fail_ep, model, ep, error_type, error_message)
                write_map_to_file(model_to_fail_ep, FAIL_MODEL_FILE)
                logger.info(model_to_fail_ep)

        os.remove(model_list_file)

    path = os.path.join(os.getcwd(), args.perf_result_path)
    if not os.path.exists(path):
        from pathlib import Path
        Path(path).mkdir(parents=True, exist_ok=True)

    if args.running_mode == "validate":
        logger.info("\n==========================================================")
        logger.info("========== Failing Models/EPs (accumulated) ==============")
        logger.info("==========================================================")

        if os.path.exists(FAIL_MODEL_FILE) or len(model_to_fail_ep) > 1:
            model_to_fail_ep = read_map_from_file(FAIL_MODEL_FILE)
            output_fail(model_to_fail_ep, os.path.join(path, benchmark_fail_csv))

            logger.info(model_to_fail_ep)


        logger.info("\n=========================================")
        logger.info("========== Models/EPs metrics  ==========")
        logger.info("=========================================")

        if os.path.exists(METRICS_FILE):
            model_to_metrics = read_map_from_file(METRICS_FILE)
            output_metrics(model_to_metrics, os.path.join(path, benchmark_metrics_csv))

    elif args.running_mode == "benchmark":
        logger.info("\n=======================================================")
        logger.info("=========== Models/EPs latency (accumulated)  ===========")
        logger.info("=======================================================")

        if os.path.exists(LATENCY_FILE):
            model_to_latency = read_map_from_file(LATENCY_FILE)
            add_improvement_information(model_to_latency)
            output_latency(model_to_latency, os.path.join(path, benchmark_latency_csv))

            pp.pprint(model_to_latency)


    logger.info("\n===========================================")
    logger.info("=========== System information  ===========")
    logger.info("===========================================")
    info = get_system_info()
    pp.pprint(info)

if __name__ == "__main__":
    main()
