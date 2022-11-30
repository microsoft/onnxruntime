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
    model_title,
    ort_provider_list,
    provider_list,
    second,
    session_name,
    session_over_time_name,
    specs_name,
    standalone_trt,
    standalone_trt_fp16,
    status_name,
    status_over_time_name,
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
    parser.add_argument("-c", "--commit_hash", help="Commit hash", required=False)
    parser.add_argument("-u", "--report_url", help="Report Url", required=True)
    parser.add_argument("-t", "--trt_version", help="Tensorrt Version", required=False)
    parser.add_argument("-b", "--branch", help="Branch", required=False)
    parser.add_argument("--kusto_conn", help="Kusto connection URL", required=True)
    parser.add_argument("--database", help="Database name", required=True)
    parser.add_argument(
        "-d",
        "--commit_datetime",
        help="Commit datetime in Python's datetime ISO 8601 format",
        required=False,
        type=datetime.datetime.fromisoformat,
        default=None,
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


def get_session_over_time(session_table):
    """
    Returns a new Pandas table with data that tracks the session creation times of model/EP combinations over time.

    :param session_table: The Pandas table containing per model/EP session creation times with the schema:
                          | Model    | ORT-CUDAFp16 | ... | ORT-CUDAFp16_second |       Group     | ...
                          =============================================================================
                          | resnet.. |     1.99     | ... |         0.92        | onnx-zoo-models | ...

    :return: A new table in which the EPs are not hardcoded as columns. Ex:
             | Model    |      Group      |      Ep      | SessionCreationTime | SessionCreationTime_second | ...
             ====================================================================================================
             | resnet.. | onnx-zoo-models | ORT-CUDAFp16 |        1.99         |             0.92           | ...
    """

    ep_names = [cpu, cuda_fp16, cuda, trt_fp16, trt]
    over_time_1 = session_table.melt(
        id_vars=[model_title, group_title], value_vars=ep_names, var_name="Ep", value_name="SessionCreationTime"
    )
    over_time_2 = session_table.melt(
        id_vars=[model_title, group_title],
        value_vars=[ep + "_second" for ep in ep_names],
        value_name="SessionCreationTime_second",
    )
    over_time = over_time_1.merge(over_time_2[["SessionCreationTime_second"]], left_index=True, right_index=True)
    over_time = over_time[
        [
            model_title,
            group_title,
            "Ep",
            "SessionCreationTime",
            "SessionCreationTime_second",
        ]
    ]

    over_time.fillna("", inplace=True)

    return over_time


def status_to_int(status):
    """
    Converts a status string to an integer.

    :param status: A status string indicating if an EP successfully inferenced a model (i.e., "Pass", "Fail", "nan")

    :return: 1 for "Pass", 0 for "Fail", and -1 otherwise.
    """

    if status == "Pass":
        return 1

    if status == "Fail":
        return 0

    return -1


def get_status_over_time(status_table):
    """
    Returns a new Pandas table with data that tracks the compatibility of model/EP combinations over time.

    :param status_table: The Pandas table containing per model/EP compatibility ('Pass' or 'Fail') data with the schema:
                          |     Model     | ORT-CUDAFp16 | ORT-CUDAFp32 | ... |       Group     | ...
                          ===========================================================================
                          | FasterRCNN-10 |     Fail     |     Pass     | ... | onnx-zoo-models | ...

    :return: A new table in which the EPs are not hardcoded as columns. Ex:
             |      Model    |      Group      |      Ep      | Pass | ...
             =============================================================
             | FasterRCNN-10 | onnx-zoo-models | ORT-CUDAFp16 |   0  | ...
             | FasterRCNN-10 | onnx-zoo-models | ORT-CUDAFp32 |   1  | ...
    """

    over_time = status_table.melt(id_vars=[model_title, group_title], var_name="Ep", value_name="Pass")
    over_time["Pass"] = over_time["Pass"].map(status_to_int)
    over_time = over_time.loc[over_time["Pass"] != -1]
    over_time = over_time[
        [
            model_title,
            group_title,
            "Ep",
            "Pass",
        ]
    ]

    return over_time


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


def get_specs(specs):
    """
    Returns a new Pandas table with data that tracks the configuration/specs/versions of the hardware and software
    used to gather benchmarking data.

    :param specs: The Pandas table containing raw specs data imported from a CSV file.

    :return: The updated table.
    """

    return specs[[".", "Spec", "Version"]]


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


def get_inference_latency(latency_table, model_group):
    latency_table = latency_table.assign(Group=model_group)
    latency_table = latency_table[
        [
            model_title,
            group_title,
            "Ep",
            "AvgLatency",
            "Latency90Pt",
            "SampleStd",
            "SampleSize",
            "UseIOBinding",
        ]
    ]

    return latency_table


def write_table(
    ingest_client, database_name, table, table_name, upload_time, identifier, branch, commit_id, commit_date
):
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
    table = table.assign(Branch=branch)
    table = table.assign(CommitId=commit_id)
    table = table.assign(CommitDate=str(commit_date))

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


def get_specs_table_value(table, spec_name):
    subtable = table.query(f"Spec == '{spec_name}'")
    vals = subtable["Version"].values

    return vals[0] if len(vals) > 0 else None


def get_build_info(csv_filenames, args):
    if not (specs_name + ".csv") in csv_filenames:
        return {
            "branch": args.branch,
            "commit_id": args.commit_hash,
            "commit_date": args.commit_datetime,
            "trt_version": args.trt_version,
        }

    specs_table = pd.read_csv(f"{specs_name}.csv")

    return {
        "branch": get_specs_table_value(specs_table, "Branch") or args.branch,
        "commit_id": get_specs_table_value(specs_table, "CommitId") or args.commit_hash,
        "commit_date": datetime.datetime.fromisoformat(get_specs_table_value(specs_table, "CommitDate"))
        or args.commit_datetime,
        "trt_version": get_specs_table_value(specs_table, "TensorRT") or args.trt_version,
    }


def main():
    """
    Entry point of this script. Uploads data produced by benchmarking scripts to the database.
    """

    args = parse_arguments()

    # connect to database
    kcsb_ingest = KustoConnectionStringBuilder.with_az_cli_authentication(args.kusto_conn)
    ingest_client = QueuedIngestClient(kcsb_ingest)
    upload_time = datetime.datetime.now(tz=datetime.timezone.utc).replace(microsecond=0)

    try:
        result_file = args.report_folder

        folders = os.listdir(result_file)
        os.chdir(result_file)

        tables = [
            fail_name,
            latency_name,
            status_name,
            status_over_time_name,
            specs_name,
            session_name,
            session_over_time_name,
        ]

        build_info = {}
        table_results = {}
        for table_name in tables:
            table_results[table_name] = pd.DataFrame()

        for model_group in folders:
            os.chdir(model_group)
            csv_filenames = os.listdir()

            if not build_info:
                build_info = get_build_info(csv_filenames, args)
                identifier = get_identifier(
                    build_info["commit_date"],
                    build_info["commit_id"],
                    build_info["trt_version"],
                    build_info["branch"],
                )

            for csv in csv_filenames:
                table = pd.read_csv(csv)
                if session_name in csv:
                    table_results[session_name] = pd.concat(
                        [table_results[session_name], get_session(table, model_group)], ignore_index=True
                    )
                elif specs_name in csv:
                    table_results[specs_name] = pd.concat(
                        [
                            table_results[specs_name],
                            get_specs(table),
                        ],
                        ignore_index=True,
                    )
                elif fail_name in csv:
                    table_results[fail_name] = pd.concat(
                        [table_results[fail_name], get_failures(table, model_group)],
                        ignore_index=True,
                    )
                elif latency_name in csv:
                    table_results[latency_name] = pd.concat(
                        [table_results[latency_name], get_inference_latency(table, model_group)],
                        ignore_index=True,
                    )
                elif status_name in csv:
                    table_results[status_name] = pd.concat(
                        [table_results[status_name], get_status(table, model_group)], ignore_index=True
                    )
            os.chdir(result_file)

        if not table_results[session_name].empty:
            table_results[session_over_time_name] = get_session_over_time(table_results[session_name])

        if not table_results[status_name].empty:
            table_results[status_over_time_name] = get_status_over_time(table_results[status_name])

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
                build_info["branch"],
                build_info["commit_id"],
                build_info["commit_date"],
            )

    except BaseException as e:
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
