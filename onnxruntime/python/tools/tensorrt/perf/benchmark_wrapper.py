import argparse  # noqa: F401
import copy  # noqa: F401
import csv  # noqa: F401
import json
import logging  # noqa: F401
import os
import pprint
import re

import coloredlogs  # noqa: F401
from benchmark import *  # noqa: F403
from perf_utils import *  # noqa: F403


def write_model_info_to_file(model, path):
    with open(path, "w") as file:
        file.write(json.dumps(model))  # use `json.loads` to do the reverse


def get_ep_list(comparison):
    if comparison == "acl":
        ep_list = [cpu, acl]  # noqa: F405
    else:
        # test with cuda and trt
        ep_list = [
            cpu,  # noqa: F405
            cuda,  # noqa: F405
            trt,  # noqa: F405
            standalone_trt,  # noqa: F405
            cuda_fp16,  # noqa: F405
            trt_fp16,  # noqa: F405
            standalone_trt_fp16,  # noqa: F405
        ]
    return ep_list


def resolve_trtexec_path(workspace):
    trtexec_options = get_output(["find", workspace, "-name", "trtexec"])  # noqa: F405
    trtexec_path = re.search(r".*/bin/trtexec", trtexec_options).group(0)
    logger.info(f"using trtexec {trtexec_path}")  # noqa: F405
    return trtexec_path


def dict_to_args(dct):
    return ",".join([f"{k}={v}" for k, v in dct.items()])


def fetch_ep_node_info(ep_node_info, model, ep):
    found_all = False
    keyphraze_all = "All nodes placed on ["
    keyphraze_each = "Node(s) placed on ["
    with open(f"VERBOSE_{model}_{ep}.log", "r") as f:
        logger.info(f"Loading VERBOSE_{model}_{ep}.log...")
        for line in f:
            # Filter out prefix
            if keyphraze_all in line:
                info = line.split(keyphraze_all)[1]
                found_all = True
            elif keyphraze_each in line:
                info = line.split(keyphraze_each)[1]
            else:
                continue
            # Parse number of nodes running on ep
            sub_ep = info.split("]")[0]
            num_of_nodes = int(info.split(": ")[-1])
            ep_node_info[model][ep][sub_ep] = num_of_nodes
            logger.info(f"{model} #node run on {sub_ep}: {num_of_nodes}")
            if found_all:
                return


def main():
    args = parse_arguments()  # noqa: F405
    setup_logger(False)  # noqa: F405
    pp = pprint.PrettyPrinter(indent=4)

    # create ep list to iterate through
    if args.ep_list:
        ep_list = args.ep_list
    else:
        ep_list = get_ep_list(args.comparison)

    trtexec = resolve_trtexec_path(args.workspace)

    models = {}
    parse_models_helper(args, models)  # noqa: F405

    model_to_fail_ep = {}
    ep_node_info = {model:{ep: {} for ep in ep_list} for model in models}

    benchmark_fail_csv = fail_name + csv_ending  # noqa: F405
    benchmark_metrics_csv = metrics_name + csv_ending  # noqa: F405
    benchmark_success_csv = success_name + csv_ending  # noqa: F405
    benchmark_latency_csv = latency_name + csv_ending  # noqa: F405
    benchmark_status_csv = status_name + csv_ending  # noqa: F405
    benchmark_session_csv = session_name + csv_ending  # noqa: F405
    specs_csv = specs_name + csv_ending  # noqa: F405

    validate = is_validate_mode(args.running_mode)  # noqa: F405
    benchmark = is_benchmark_mode(args.running_mode)  # noqa: F405

    for model, model_info in models.items():
        logger.info("\n" + "=" * 40 + "=" * len(model))  # noqa: F405
        logger.info("=" * 20 + model + "=" * 20)  # noqa: F405
        logger.info("=" * 40 + "=" * len(model))  # noqa: F405

        model_info["model_name"] = model

        model_list_file = os.path.join(os.getcwd(), model + ".json")
        write_model_info_to_file([model_info], model_list_file)

        for ep in ep_list:
            command = [
                "python3",
                "benchmark.py",
                "-r",
                args.running_mode,
                "-m",
                model_list_file,
                "-o",
                args.perf_result_path,
                "--ep",
                ep,
                "--write_test_result",
                "false",
            ]

            if args.track_memory:
                command.append("-z")

            if ep in (standalone_trt, standalone_trt_fp16):  # noqa: F405
                command.extend(["--trtexec", trtexec])

            if len(args.cuda_ep_options):
                command.extend(["--cuda_ep_options", dict_to_args(args.cuda_ep_options)])

            if len(args.trt_ep_options):
                command.extend(["--trt_ep_options", dict_to_args(args.trt_ep_options)])

            if validate:
                command.extend(["--benchmark_metrics_csv", benchmark_metrics_csv])

            if benchmark:
                command.extend(
                    [
                        "-t",
                        str(args.test_times),
                        "-o",
                        args.perf_result_path,
                        "--write_test_result",
                        "false",
                        "--benchmark_fail_csv",
                        benchmark_fail_csv,
                        "--benchmark_latency_csv",
                        benchmark_latency_csv,
                        "--benchmark_success_csv",
                        benchmark_success_csv,
                    ]
                )

            with open(f"VERBOSE_{model}_{ep}.log", "w") as f:
                p = subprocess.run(command, stdout=f, stderr=subprocess.PIPE)

            logger.info("Completed subprocess %s ", " ".join(p.args))  # noqa: F405
            logger.info("Return code: %d", p.returncode)  # noqa: F405

            if p.returncode != 0:
                error_type = "runtime error"
                error_message = "Benchmark script exited with returncode = " + str(p.returncode)

                if p.stderr:
                    error_message += "\nSTDERR:\n" + p.stderr.decode("utf-8")

                logger.error(error_message)  # noqa: F405
                update_fail_model_map(model_to_fail_ep, model, ep, error_type, error_message)  # noqa: F405
                write_map_to_file(model_to_fail_ep, FAIL_MODEL_FILE)  # noqa: F405
                logger.info(model_to_fail_ep)  # noqa: F405

            fetch_ep_node_info(ep_node_info, model, ep)

        os.remove(model_list_file)

    path = os.path.join(os.getcwd(), args.perf_result_path)
    if not os.path.exists(path):
        from pathlib import Path

        Path(path).mkdir(parents=True, exist_ok=True)

    if validate:
        logger.info("\n=========================================")  # noqa: F405
        logger.info("=========== Models/EPs metrics ==========")  # noqa: F405
        logger.info("=========================================")  # noqa: F405

        if os.path.exists(METRICS_FILE):  # noqa: F405
            model_to_metrics = read_map_from_file(METRICS_FILE)  # noqa: F405
            output_metrics(model_to_metrics, os.path.join(path, benchmark_metrics_csv), ep_node_info)  # noqa: F405
            logger.info(f"\nSaved model metrics results to {benchmark_metrics_csv}")  # noqa: F405

    if benchmark:
        logger.info("\n=========================================")  # noqa: F405
        logger.info("======= Models/EPs session creation =======")  # noqa: F405
        logger.info("=========================================")  # noqa: F405

        if os.path.exists(SESSION_FILE):  # noqa: F405
            model_to_session = read_map_from_file(SESSION_FILE)  # noqa: F405
            pretty_print(pp, model_to_session)  # noqa: F405
            output_session_creation(model_to_session, os.path.join(path, benchmark_session_csv))  # noqa: F405
            logger.info(f"\nSaved session creation results to {benchmark_session_csv}")  # noqa: F405

        logger.info("\n=========================================================")  # noqa: F405
        logger.info("========== Failing Models/EPs (accumulated) ==============")  # noqa: F405
        logger.info("==========================================================")  # noqa: F405

        if os.path.exists(FAIL_MODEL_FILE) or len(model_to_fail_ep) > 1:  # noqa: F405
            model_to_fail_ep = read_map_from_file(FAIL_MODEL_FILE)  # noqa: F405
            output_fail(model_to_fail_ep, os.path.join(path, benchmark_fail_csv))  # noqa: F405
            logger.info(model_to_fail_ep)  # noqa: F405
            logger.info(f"\nSaved model failing results to {benchmark_fail_csv}")  # noqa: F405

        logger.info("\n=======================================================")  # noqa: F405
        logger.info("=========== Models/EPs Status (accumulated) ===========")  # noqa: F405
        logger.info("=======================================================")  # noqa: F405

        model_status = {}
        if os.path.exists(LATENCY_FILE):  # noqa: F405
            model_latency = read_map_from_file(LATENCY_FILE)  # noqa: F405
            is_fail = False
            model_status = build_status(model_status, model_latency, is_fail)  # noqa: F405
        if os.path.exists(FAIL_MODEL_FILE):  # noqa: F405
            model_fail = read_map_from_file(FAIL_MODEL_FILE)  # noqa: F405
            is_fail = True
            model_status = build_status(model_status, model_fail, is_fail)  # noqa: F405

        pretty_print(pp, model_status)  # noqa: F405

        output_status(model_status, os.path.join(path, benchmark_status_csv))  # noqa: F405
        logger.info(f"\nSaved model status results to {benchmark_status_csv}")  # noqa: F405

        logger.info("\n=========================================================")  # noqa: F405
        logger.info("=========== Models/EPs latency (accumulated)  ===========")  # noqa: F405
        logger.info("=========================================================")  # noqa: F405

        if os.path.exists(LATENCY_FILE):  # noqa: F405
            model_to_latency = read_map_from_file(LATENCY_FILE)  # noqa: F405
            add_improvement_information(model_to_latency)  # noqa: F405

            pretty_print(pp, model_to_latency)  # noqa: F405

            output_latency(model_to_latency, os.path.join(path, benchmark_latency_csv))  # noqa: F405
            logger.info(f"\nSaved model latency results to {benchmark_latency_csv}")  # noqa: F405

    logger.info("\n===========================================")  # noqa: F405
    logger.info("=========== System information  ===========")  # noqa: F405
    logger.info("===========================================")  # noqa: F405
    info = get_system_info(args)  # noqa: F405
    pretty_print(pp, info)  # noqa: F405
    logger.info("\n")  # noqa: F405
    output_specs(info, os.path.join(path, specs_csv))  # noqa: F405
    logger.info(f"\nSaved hardware specs to {specs_csv}")  # noqa: F405


if __name__ == "__main__":
    main()
