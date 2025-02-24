# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import csv
import datetime
import os
import re

import pandas as pd
from azure.kusto.data import KustoConnectionStringBuilder
from azure.kusto.ingest import QueuedIngestClient
from post import get_identifier, parse_arguments, write_table


def parse_valgrind_log(input_path, output_path, keywords):
    is_definitely_lost = False
    is_ort_trt_related = False
    buffer = []
    leak_block = None
    leak_bytes = None
    keyword = None
    results = []

    with open(input_path) as file:
        lines = file.readlines()

        for line in lines:
            line = line.strip()  # noqa: PLW2901
            # Remove "==xxxxx==" pattern from the line
            line = line.split("==")[-1].strip()  # noqa: PLW2901

            if "blocks are definitely lost in loss" in line:
                is_definitely_lost = True
                # Extract LeakBlock and LeakBytes
                match = re.search(r"([\d,]+) byte[s]? in ([\d,]+) block[s]?", line)
                if match:
                    leak_bytes = match.group(1).replace(",", "")
                    leak_block = match.group(2).replace(",", "")
                continue

            if is_definitely_lost:
                if line:
                    buffer.append(line)
                    for word in keywords:
                        if word in line:
                            is_ort_trt_related = True
                            keyword = word
                            break

            # End of section
            if is_definitely_lost and not line:
                if is_ort_trt_related:
                    results.append((keyword, leak_block, leak_bytes, "\n".join(buffer)))
                # Reset var
                is_definitely_lost = False
                is_ort_trt_related = False
                buffer = []
                leak_block = None
                leak_bytes = None
                keyword = None

    # Writing results to CSV
    with open(output_path, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Keyword", "LeakBlock", "LeakBytes", "ValgrindMessage"])
        for entry in results:
            csvwriter.writerow([entry[0], entry[1], entry[2], entry[3]])


def parse_concurrency_test_log(input_path, output_path):
    with open(input_path) as log_file:
        log_content = log_file.read()

    failed_cases_section = log_content.split("Failed Test Cases:")[1]

    # passed = 1 if no failed test cases
    if failed_cases_section.strip() == "":
        passed = 1
    else:
        passed = 0

    with open(output_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Passed", "Log"])
        csv_writer.writerow([passed, log_content])


if __name__ == "__main__":
    args = parse_arguments()

    # connect to database
    kcsb_ingest = KustoConnectionStringBuilder.with_az_cli_authentication(args.kusto_conn)
    ingest_client = QueuedIngestClient(kcsb_ingest)
    identifier = get_identifier(
        args.commit_datetime, args.commit_hash, args.trt_version, args.branch, args.use_tensorrt_oss_parser
    )
    upload_time = datetime.datetime.now(tz=datetime.timezone.utc).replace(microsecond=0)

    try:
        result_mem_test_path = args.report_folder
        os.chdir(result_mem_test_path)
        # Parse mem_test log
        logs = ["valgrind.log", "concurrency_test.log"]
        csv_paths = ["mem_test.csv", "concurrency_test.csv"]
        for log, csv_path in zip(logs, csv_paths, strict=False):
            if os.path.exists(log):
                print(f"{identifier}: Parsing {log}")
                if log == logs[0]:
                    parse_valgrind_log(log, csv_path, ["TensorrtExecutionProvider", "TensorRT"])
                else:
                    parse_concurrency_test_log(log, csv_path)

        # Upload to db
        for csv_path, db_table_name in zip(
            csv_paths, ["ep_valgrind_record", "ep_concurrencytest_record"], strict=False
        ):
            if os.path.exists(csv_path):
                table = pd.read_csv(csv_path)
                write_table(
                    ingest_client,
                    args.database,
                    table,
                    db_table_name,
                    upload_time,
                    identifier,
                    args.branch,
                    args.commit_hash,
                    args.commit_datetime,
                )
                print(f"{identifier}: {csv_path} is synced to db")

    except Exception as e:
        print(str(e))
