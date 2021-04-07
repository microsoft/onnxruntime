import os
import csv
import logging
import coloredlogs
import argparse
import copy
import json
import re
import pprint
from benchmark import *

def write_model_info_to_file(model, path):
    with open(path, 'w') as file:
        file.write(json.dumps(model)) # use `json.loads` to do the reverse

def get_ep_list(comparison): 
    if comparison == 'acl': 
        ep_list = [cpu, acl]
    else:   
        # test with cuda and trt
        ep_list = [cpu, cuda, trt, standalone_trt, cuda_fp16, trt_fp16, standalone_trt_fp16]
    return ep_list

def main():
    args = parse_arguments()
    setup_logger(False)
    pp = pprint.PrettyPrinter(indent=4)

    models = {}
    parse_models_helper(args, models)

    model_to_fail_ep = {}

    benchmark_fail_csv = 'fail.csv'
    benchmark_metrics_csv = 'metrics.csv' 
    benchmark_success_csv = 'success.csv'  
    benchmark_latency_csv = 'latency.csv' 
    benchmark_status_csv = 'status.csv' 

    for model, model_info in models.items():
        logger.info("\n" + "="*40 + "="*len(model))
        logger.info("="*20 + model +"="*20)
        logger.info("="*40 + "="*len(model))

        model_info["model_name"] = model 
        
        model_list_file = os.path.join(os.getcwd(), model +'.json')
        write_model_info_to_file([model_info], model_list_file)
        
        if args.ep: 
            ep_list = [args.ep]
        else:
            ep_list = get_ep_list(args.comparison)
        
        for ep in ep_list:
            
            command =  ["python3",
                        "benchmark.py",
                        "-r", args.running_mode,
                        "-m", model_list_file,
                        "-o", args.perf_result_path,
                        "--write_test_result", "false"]
            
            if "Standalone" in ep: 
                if args.running_mode == "validate": 
                    continue 
                else:
                    trtexec_path = get_trtexec_path()    
                    command.extend(["--trtexec", trtexec_path])
                    ep = trt_fp16 if "fp16" in ep else trt 

            command.extend(["--ep", ep])
            
            if args.running_mode == "validate":
                command.extend(["--benchmark_fail_csv", benchmark_fail_csv,
                                "--benchmark_metrics_csv", benchmark_metrics_csv])
            
            elif args.running_mode == "benchmark":
                command.extend(["-t", str(args.test_times),
                                "-o", args.perf_result_path,
                                "--write_test_result", "false",
                                "--benchmark_latency_csv", benchmark_latency_csv,
                                "--benchmark_success_csv", benchmark_success_csv]) 
            
            p = subprocess.run(command)
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
        logger.info("\n=========================================================")
        logger.info("========== Failing Models/EPs (accumulated) ==============")
        logger.info("==========================================================")

        if os.path.exists(FAIL_MODEL_FILE) or len(model_to_fail_ep) > 1:
            model_to_fail_ep = read_map_from_file(FAIL_MODEL_FILE)
            output_fail(model_to_fail_ep, os.path.join(path, benchmark_fail_csv))
            logger.info("\nSaved model fail results to {}".format(benchmark_fail_csv)) 
            logger.info(model_to_fail_ep)

        logger.info("\n=========================================")
        logger.info("=========== Models/EPs metrics ==========")
        logger.info("=========================================")

        if os.path.exists(METRICS_FILE):
            model_to_metrics = read_map_from_file(METRICS_FILE)
            output_metrics(model_to_metrics, os.path.join(path, benchmark_metrics_csv))
            logger.info("\nSaved model metrics results to {}".format(benchmark_metrics_csv)) 
    
    elif args.running_mode == "benchmark":
        logger.info("\n=======================================================")
        logger.info("=========== Models/EPs Status (accumulated) ===========")
        logger.info("=======================================================")

        model_status = {}
        if os.path.exists(LATENCY_FILE):
            model_latency = read_map_from_file(LATENCY_FILE)
            is_fail = False
            model_status = build_status(model_status, model_latency, is_fail)
        if os.path.exists(FAIL_MODEL_FILE):
            model_fail = read_map_from_file(FAIL_MODEL_FILE)
            is_fail = True
            model_status = build_status(model_status, model_fail, is_fail)
       
        pretty_print(pp, model_status)
        
        output_status(model_status, os.path.join(path, benchmark_status_csv)) 
        logger.info("\nSaved model status results to {}".format(benchmark_status_csv)) 

        logger.info("\n=========================================================")
        logger.info("=========== Models/EPs latency (accumulated)  ===========")
        logger.info("=========================================================")

        if os.path.exists(LATENCY_FILE):
            model_to_latency = read_map_from_file(LATENCY_FILE)
            add_improvement_information(model_to_latency)
            
            pretty_print(pp, model_to_latency)
            
            output_latency(model_to_latency, os.path.join(path, benchmark_latency_csv))
            logger.info("\nSaved model status results to {}".format(benchmark_latency_csv)) 

    logger.info("\n===========================================")
    logger.info("=========== System information  ===========")
    logger.info("===========================================")
    info = get_system_info()
    pretty_print(pp, info)

if __name__ == "__main__":
    main()
