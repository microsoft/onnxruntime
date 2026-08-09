#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


# command line arguments
# --report_url=<string>
# --report_file=<string, local file path, TXT/JSON file>
# --commit_hash=<string, full git commit hash>

import argparse
import datetime
import json
import sys

# ingest from dataframe
import pandas
from azure.kusto.data import DataFormat, KustoConnectionStringBuilder
from azure.kusto.ingest import IngestionProperties, QueuedIngestClient, ReportLevel


def parse_arguments():
    parser = argparse.ArgumentParser(description="ONNXRuntime test coverage report uploader for dashboard")
    parser.add_argument("--report_url", type=str, help="URL to the LLVM json report")
    parser.add_argument("--report_file", type=str, help="Path to the local JSON/TXT report", required=True)
    parser.add_argument("--commit_hash", type=str, help="Full Git commit hash", required=True)
    parser.add_argument("--branch", type=str, help="Source code branch")
    parser.add_argument("--os", type=str, help="Build configuration:os")
    parser.add_argument("--arch", type=str, help="Build configuration:arch")
    parser.add_argument("--build_config", type=str, help="Build configuration: build variants")
    return parser.parse_args()


def parse_txt_report(report_file):
    data = {}
    with open(report_file) as report:
        for line in reversed(report.readlines()):
            if "TOTAL" in line:
                fields = line.strip().split()
                data["lines_valid"] = int(fields[1])
                data["lines_covered"] = int(fields[2])
                data["coverage"] = float(fields[3].strip("%")) / 100
                break
    return data


def parse_json_report(report_file):
    result = {}
    with open(report_file) as json_file:
        data = json.load(json_file)

    linestat = data["data"][0]["totals"]["lines"]
    result["coverage"] = float(linestat["percent"] / 100.0)
    result["lines_covered"] = int(linestat["covered"])
    result["lines_valid"] = int(linestat["count"])
    return result


def write_to_db(coverage_data, args):
    # connect to database
    cluster = "https://ingest-onnxruntimedashboarddb.southcentralus.kusto.windows.net"
    kcsb = KustoConnectionStringBuilder.with_az_cli_authentication(cluster)
    # The authentication method will be taken from the chosen KustoConnectionStringBuilder.
    client = QueuedIngestClient(kcsb)
    fields = [
        "UploadTime",
        "CommitId",
        "Coverage",
        "LinesCovered",
        "TotalLines",
        "OS",
        "Arch",
        "BuildConfig",
        "ReportURL",
        "Branch",
    ]
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = [
        [
            now_str,
            args.commit_hash,
            coverage_data["coverage"],
            coverage_data["lines_covered"],
            coverage_data["lines_valid"],
            args.os.lower(),
            args.arch.lower(),
            args.build_config.lower(),
            args.report_url.lower(),
            args.branch.lower(),
        ]
    ]
    ingestion_props = IngestionProperties(
        database="powerbi",
        table="test_coverage",
        data_format=DataFormat.CSV,
        report_level=ReportLevel.FailuresAndSuccesses,
    )
    df = pandas.DataFrame(data=rows, columns=fields)
    client.ingest_from_dataframe(df, ingestion_properties=ingestion_props)


if __name__ == "__main__":
    try:
        args = parse_arguments()
        if args.report_file.endswith(".json"):
            coverage_data = parse_json_report(args.report_file)
        elif args.report_file.endswith(".txt"):
            coverage_data = parse_txt_report(args.report_file)
        else:
            raise ValueError("Only report extensions txt or json are accepted")

        write_to_db(coverage_data, args)
    except BaseException as e:
        print(str(e))
        sys.exit(1)
