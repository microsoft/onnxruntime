#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import argparse
import datetime
import os
import sys

# ingest from dataframe
import pandas
from azure.kusto.data import DataFormat, KustoConnectionStringBuilder
from azure.kusto.ingest import IngestionProperties, QueuedIngestClient, ReportLevel


def parse_arguments():
    parser = argparse.ArgumentParser(description="ONNXRuntime binary size uploader for dashboard")
    parser.add_argument("--commit_hash", help="Full Git commit hash")
    parser.add_argument(
        "--build_project",
        default="Lotus",
        choices=["Lotus", "onnxruntime"],
        help="Lotus or onnxruntime build project, to construct the build URL",
    )
    parser.add_argument("--build_id", help="Build Id")
    parser.add_argument("--size_data_file", help="Path to file that contains the binary size data")
    parser.add_argument(
        "--ignore_db_error", action="store_true", help="Ignore database errors while executing this script"
    )

    return parser.parse_args()


# Assumes size_data_file is a csv file with a header line, containing binary sizes and other attributes
# CSV fields are:
#    os,arch,build_config,size
# No empty line or space between fields expected


def get_binary_sizes(size_data_file):
    binary_size = []
    with open(size_data_file) as f:
        line = f.readline()
        headers = line.strip().split(",")
        while line:
            line = f.readline()
            if not line:
                break
            linedata = line.strip().split(",")
            tablerow = {}
            for i in range(len(headers)):
                if headers[i] == "size":
                    tablerow[headers[i]] = int(linedata[i])
                else:
                    tablerow[headers[i]] = linedata[i]
            binary_size.append(tablerow)
    return binary_size


def write_to_db(binary_size_data, args):
    # connect to database
    cluster = "https://ingest-onnxruntimedashboarddb.southcentralus.kusto.windows.net"
    kcsb = KustoConnectionStringBuilder.with_az_cli_authentication(cluster)
    # The authentication method will be taken from the chosen KustoConnectionStringBuilder.
    client = QueuedIngestClient(kcsb)
    fields = ["build_time", "build_id", "build_project", "commit_id", "os", "arch", "build_config", "size", "Branch"]
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    branch_name = os.environ.get("BUILD_SOURCEBRANCHNAME", "main")
    rows = []
    for row in binary_size_data:
        rows.append(
            [
                now_str,
                args.build_id,
                args.build_project,
                args.commit_hash,
                row["os"],
                row["arch"],
                row["build_config"],
                row["size"],
                branch_name.lower(),
            ]
        )
    ingestion_props = IngestionProperties(
        database="powerbi",
        table="binary_size",
        data_format=DataFormat.CSV,
        report_level=ReportLevel.FailuresAndSuccesses,
    )
    df = pandas.DataFrame(data=rows, columns=fields)
    client.ingest_from_dataframe(df, ingestion_properties=ingestion_props)


if __name__ == "__main__":
    args = parse_arguments()
    binary_size_data = get_binary_sizes(args.size_data_file)
    try:
        write_to_db(binary_size_data, args)
    except BaseException as e:
        print(str(e))
        # If there is DB connection error, and we choose '--ignore_db_error'
        # we can let the script exit clean in order not to fail the pipeline
        if not args.ignore_db_error:
            sys.exit(1)
