# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import datetime
import os
import sys

import pandas as pd
from azure.kusto.data import KustoConnectionStringBuilder
from azure.kusto.data.data_format import DataFormat
from azure.kusto.ingest import IngestionProperties, QueuedIngestClient, ReportLevel
from perf_utils import (
    avg_ending,
    cpu,
    cuda,
    cuda_fp16,
    fail_name,
    group_title,
    latency_name,
    latency_over_time_name,
    memory_ending,
    memory_name,
    model_title,
    ort_provider_list,
    provider_list,
    second,
    session_name,
    specs_name,
    standalone_trt,
    standalone_trt_fp16,
    status_name,
    table_headers,
    trt,
    trt_fp16,
)


def parse_arguments():
    """
    Parses command-line arguments and returns an object with each argument as a field.

    :return: An object whose fields represent the parsed command-line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--report_folder", help="Path to the local file report", required=True)
    parser.add_argument("-c", "--commit_hash", help="Commit hash", required=True)
    parser.add_argument("-u", "--report_url", help="Report Url", required=True)
    parser.add_argument("-t", "--trt_version", help="Tensorrt Version", required=True)
    parser.add_argument("-b", "--branch", help="Branch", required=True)
    parser.add_argument("--kusto_conn", help="Kusto connection URL", required=True)
    parser.add_argument("--database", help="Database name", required=True)
    parser.add_argument(
        "-d",
        "--commit_datetime",
        help="Commit datetime in Python's datetime ISO 8601 format",
        required=True,
        type=datetime.datetime.fromisoformat,
    )

    return parser.parse_args()


def adjust_columns(table, columns, db_columns, model_group):
    """
    Utility function that replaces column names in an in-memory table with the appropriate database column names.
    Additionly, this function adds a model group column to all rows in the table.

    :param table: The Pandas table to adjust.
    :param columns: A list of existing column names to rename.
    :param db_columns: A list of databse columns names to use.
    :param model_group: The model group to append as a column.

    :return: The updated table.
    """

    table = table[columns]
    table = table.set_axis(db_columns, axis=1)
    table = table.assign(Group=model_group)
    return table


def get_latency_over_time(commit_hash, report_url, branch, latency_table):
    """
    Returns a new Pandas table with data that tracks the latency of model/EP inference runs over time.

    :param commit_hash: The short git commit hash corresponding to the version of ORT used to gather latency data.
    :param report_url: The URL of the Azure pipeline run/report which produced this latency data.
    :param branch: The name of the git branch corresponding to the version of ORT used to gather latency data.
    :param latency_table: The Pandas table containing raw "latency over time" data imported from a CSV file.

    :return: The updated table.
    """

    over_time = latency_table
    over_time = over_time.melt(id_vars=[model_title, group_title], var_name="Ep", value_name="Latency")
    over_time = over_time.assign(CommitId=commit_hash)
    over_time = over_time.assign(ReportUrl=report_url)
    over_time = over_time.assign(Branch=branch)
    over_time = over_time[
        [
            "CommitId",
            model_title,
            "Ep",
            "Latency",
            "ReportUrl",
            group_title,
            "Branch",
        ]
    ]
    over_time.fillna("", inplace=True)
    return over_time


def get_failures(fail, model_group):
    """
    Returns a new Pandas table with data that tracks failed model/EP inference runs.

    :param fail: The Pandas table containing raw failure data imported from a CSV file.
    :param model_group: The model group namespace to append as a column.

    :return: The updated table.
    """

    fail_columns = fail.keys()
    fail_db_columns = [model_title, "Ep", "ErrorType", "ErrorMessage"]
    fail = adjust_columns(fail, fail_columns, fail_db_columns, model_group)
    return fail


def get_memory(memory, model_group):
    """
    Returns a new Pandas table with data that tracks peak memory usage per model/EP.

    :param memory: The Pandas table containing raw memory usage data imported from a CSV file.
    :param model_group: The model group namespace to append as a column.

    :return: The updated table.
    """

    memory_columns = [model_title]
    for provider in provider_list:
        if cpu not in provider:
            memory_columns.append(provider + memory_ending)
    memory_db_columns = [
        model_title,
        cuda,
        trt,
        standalone_trt,
        cuda_fp16,
        trt_fp16,
        standalone_trt_fp16,
    ]
    memory = adjust_columns(memory, memory_columns, memory_db_columns, model_group)
    return memory


def get_latency(latency, model_group):
    """
    Returns a new Pandas table with data that tracks inference run latency per model/EP.

    :param latency: The Pandas table containing raw latency data imported from a CSV file.
    :param model_group: The model group namespace to append as a column.

    :return: The updated table.
    """

    latency_columns = [model_title]
    for provider in provider_list:
        latency_columns.append(provider + avg_ending)
    latency_db_columns = table_headers
    latency = adjust_columns(latency, latency_columns, latency_db_columns, model_group)
    return latency


def get_status(status, model_group):
    """
    Returns a new Pandas table with data that tracks whether an EP can successfully run a particular model.

    :param status: The Pandas table containing raw model/EP status data imported from a CSV file.
    :param model_group: The model group namespace to append as a column.

    :return: The updated table.
    """

    status_columns = status.keys()
    status_db_columns = table_headers
    status = adjust_columns(status, status_columns, status_db_columns, model_group)
    return status


def get_specs(specs, branch, commit_hash, commit_datetime):
    """
    Returns a new Pandas table with data that tracks the configuration/specs/versions of the hardware and software
    used to gather benchmarking data.

    :param specs: The Pandas table containing raw specs data imported from a CSV file.
    :param branch: The name of the git branch corresponding to the version of ORT used to gather data.
    :param commit_hash: The short git commit hash corresponding to the version of ORT used to gather data.
    :param commit_datetime: The git commit datetime corresponding to the version of ORT used to gather data.

    :return: The updated table.
    """

    init_id = int(specs.tail(1).get(".", 0)) + 1
    specs_additional = pd.DataFrame(
        {
            ".": [init_id, init_id + 1, init_id + 2],
            "Spec": ["Branch", "CommitId", "CommitTime"],
            "Version": [branch, commit_hash, str(commit_datetime)],
        }
    )

    return pd.concat([specs, specs_additional], ignore_index=True)


def get_session(session, model_group):
    """
    Returns a new Pandas table with data that tracks the ORT session creation time for each model/EP combination.

    :param session: The Pandas table containing raw model/EP session timing data imported from a CSV file.
    :param model_group: The model group namespace to append as a column.

    :return: The updated table.
    """

    session_columns = session.keys()
    session_db_columns = [model_title] + ort_provider_list + [p + second for p in ort_provider_list]
    session = adjust_columns(session, session_columns, session_db_columns, model_group)
    return session


def write_table(ingest_client, database_name, table, table_name, upload_time, identifier):
    """
    Uploads the provided table to the database. This function also appends the upload time and unique run identifier
    to the table.

    :param ingest_client: An instance of QueuedIngestClient used to initiate data ingestion.
    :param table: The Pandas table to ingest.
    :param table_name: The name of the table in the database.
    :param upload_time: A datetime object denoting the data's upload time.
    :param identifier: An identifier that associates the uploaded data with an ORT commit/date/branch.
    """

    if table.empty:
        return

    # Add upload time and identifier columns to data table.
    table = table.assign(UploadTime=str(upload_time))
    table = table.assign(Identifier=identifier)
    ingestion_props = IngestionProperties(
        database=database_name,
        table=table_name,
        data_format=DataFormat.CSV,
        report_level=ReportLevel.FailuresAndSuccesses,
    )
    # append rows
    ingest_client.ingest_from_dataframe(table, ingestion_properties=ingestion_props)


def get_identifier(commit_datetime, commit_hash, trt_version, branch):
    """
    Returns an identifier that associates uploaded data with an ORT commit/date/branch and a TensorRT version.

    :param commit_datetime: The datetime of the ORT commit used to run the benchmarks.
    :param commit_hash: The hash of the ORT commit used to run the benchmarks.
    :param trt_version: The TensorRT version used to run the benchmarks.
    :param branch: The name of the ORT branch used to run the benchmarks.

    :return: A string identifier.
    """

    date = str(commit_datetime.date())  # extract date only
    return date + "_" + commit_hash + "_" + trt_version + "_" + branch


def main():
    """
    Entry point of this script. Uploads data produced by benchmarking scripts to the database.
    """

    args = parse_arguments()

    # connect to database
    kcsb_ingest = KustoConnectionStringBuilder.with_az_cli_authentication(args.kusto_conn)
    ingest_client = QueuedIngestClient(kcsb_ingest)
    identifier = get_identifier(args.commit_datetime, args.commit_hash, args.trt_version, args.branch)
    upload_time = datetime.datetime.now(tz=datetime.timezone.utc).replace(microsecond=0)

    try:
        result_file = args.report_folder

        folders = os.listdir(result_file)
        os.chdir(result_file)

        tables = [
            fail_name,
            memory_name,
            latency_name,
            status_name,
            latency_over_time_name,
            specs_name,
            session_name,
        ]
        table_results = {}
        for table_name in tables:
            table_results[table_name] = pd.DataFrame()

        for model_group in folders:
            os.chdir(model_group)
            csv_filenames = os.listdir()
            for csv in csv_filenames:
                table = pd.read_csv(csv)
                if session_name in csv:
                    table_results[session_name] = table_results[session_name].append(
                        get_session(table, model_group), ignore_index=True
                    )
                elif specs_name in csv:
                    table_results[specs_name] = table_results[specs_name].append(
                        get_specs(table, args.branch, args.commit_hash, args.commit_datetime),
                        ignore_index=True,
                    )
                elif fail_name in csv:
                    table_results[fail_name] = table_results[fail_name].append(
                        get_failures(table, model_group), ignore_index=True
                    )
                elif latency_name in csv:
                    table_results[memory_name] = table_results[memory_name].append(
                        get_memory(table, model_group), ignore_index=True
                    )
                    table_results[latency_name] = table_results[latency_name].append(
                        get_latency(table, model_group), ignore_index=True
                    )
                    if not table_results[latency_name].empty:
                        table_results[latency_over_time_name] = table_results[latency_over_time_name].append(
                            get_latency_over_time(
                                args.commit_hash,
                                args.report_url,
                                args.branch,
                                table_results[latency_name],
                            ),
                            ignore_index=True,
                        )
                elif status_name in csv:
                    table_results[status_name] = table_results[status_name].append(
                        get_status(table, model_group), ignore_index=True
                    )
            os.chdir(result_file)
        for table in tables:
            print("writing " + table + " to database")
            db_table_name = "ep_model_" + table
            write_table(
                ingest_client,
                args.database,
                table_results[table],
                db_table_name,
                upload_time,
                identifier,
            )

    except BaseException as e:
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
