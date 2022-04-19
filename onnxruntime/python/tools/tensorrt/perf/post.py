import argparse
import sys
import os
import pandas as pd
import time
from azure.kusto.data import KustoConnectionStringBuilder
from azure.kusto.data.data_format import DataFormat
from azure.kusto.data.helpers import dataframe_from_result_table 
from azure.kusto.ingest import (
    IngestionProperties,
    ReportLevel,
    QueuedIngestClient,
)
from perf_utils import *

# database connection strings 
cluster_ingest = "https://ingest-onnxruntimedashboarddb.southcentralus.kusto.windows.net"
database = "ep_perf_dashboard"

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", "--report_folder", help="Path to the local file report", required=True)
    parser.add_argument(
        "-c", "--commit_hash", help="Commit id", required=True)
    parser.add_argument(
        "-u", "--report_url", help="Report Url", required=True)
    parser.add_argument(
        "-t", "--trt_version", help="Tensorrt Version", required=True)
    parser.add_argument(
        "-b", "--branch", help="Branch", required=True)
    parser.add_argument(
        "-d", "--datetime", help="Commit Datetime", required=True)
    return parser.parse_args()

def parse_csv(report_file):
    table = pd.read_csv(report_file)
    return table

def adjust_columns(table, columns, db_columns, model_group): 
    table = table[columns]
    table = table.set_axis(db_columns, axis=1)
    table = table.assign(Group=model_group)
    return table 

def get_latency_over_time(commit_hash, report_url, branch, latency_table):
    if not latency_table.empty:
        over_time = latency_table
        over_time = over_time.melt(id_vars=[model_title, group_title], var_name='Ep', value_name='Latency')
        over_time = over_time.assign(CommitId=commit_hash)
        over_time = over_time.assign(ReportUrl=report_url)
        over_time = over_time.assign(Branch=branch)
        over_time = over_time[['CommitId', model_title, 'Ep', 'Latency', 'ReportUrl', group_title, 'Branch']]
        over_time.fillna('', inplace=True)
        return over_time
    
def get_failures(fail, model_group):
    fail_columns = fail.keys()
    fail_db_columns = [model_title, 'Ep', 'ErrorType', 'ErrorMessage']
    fail = adjust_columns(fail, fail_columns, fail_db_columns, model_group)
    return fail

def get_memory(memory, model_group): 
    memory_columns = [model_title]
    for provider in provider_list: 
        if cpu not in provider:
            memory_columns.append(provider + memory_ending)
    memory_db_columns = [model_title, cuda, trt, standalone_trt, cuda_fp16, trt_fp16, standalone_trt_fp16]
    memory = adjust_columns(memory, memory_columns, memory_db_columns, model_group)
    return memory

def get_latency(latency, model_group):
    latency_columns = [model_title]
    for provider in provider_list: 
        latency_columns.append(provider + avg_ending)
    latency_db_columns = table_headers
    latency = adjust_columns(latency, latency_columns, latency_db_columns, model_group)
    return latency
    
def get_status(status, model_group):
    status_columns = status.keys()
    status_db_columns = table_headers
    status = adjust_columns(status, status_columns, status_db_columns, model_group)
    return status

def get_specs(specs, branch, commit_id, date_time):
    specs = specs.append({'.': 6, 'Spec': 'Branch', 'Version' : branch}, ignore_index=True)
    specs = specs.append({'.': 7, 'Spec': 'CommitId', 'Version' : commit_id}, ignore_index=True)
    specs = specs.append({'.': 8, 'Spec': 'CommitTime', 'Version' : date_time}, ignore_index=True)
    return specs

def get_session(session, model_group):
    session_columns = session.keys()
    session_db_columns = [model_title] + ort_provider_list + [p + second for p in ort_provider_list]
    session = adjust_columns(session, session_columns, session_db_columns, model_group)
    return session

def write_table(ingest_client, table, table_name, commit_time, identifier):
    if table.empty:
        return
    table = table.assign(UploadTime=commit_time) # add Commit DateTime
    table = table.assign(Identifier=identifier) # add Identifier
    ingestion_props = IngestionProperties(
      database=database,
      table=table_name,
      data_format=DataFormat.CSV,
      report_level=ReportLevel.FailuresAndSuccesses
    )
    # append rows
    ingest_client.ingest_from_dataframe(table, ingestion_properties=ingestion_props)

def get_time():   
    date_time = time.strftime(time_string_format)
    return date_time

def get_identifier(date_time, commit_id, trt_version, branch):
    date = date_time.split('T')[0] # extract date only 
    return date + '_' + commit_id + '_' + trt_version + '_' + branch

def main():
    
    args = parse_arguments()
    
    # connect to database
    kcsb_ingest = KustoConnectionStringBuilder.with_az_cli_authentication(cluster_ingest)
    ingest_client = QueuedIngestClient(kcsb_ingest)
    date_time = args.datetime
    identifier = get_identifier(date_time, args.commit_hash, args.trt_version, args.branch)
    
    try:
        result_file = args.report_folder

        folders = os.listdir(result_file)
        os.chdir(result_file)

        tables = [fail_name, memory_name, latency_name, status_name, latency_over_time_name, specs_name, session_name]
        table_results = {}
        for table_name in tables:
            table_results[table_name] = pd.DataFrame()

        for model_group in folders:
            os.chdir(model_group)
            csv_filenames = os.listdir()
            for csv in csv_filenames:
                table = parse_csv(csv)
                if session_name in csv: 
                    table_results[session_name] = table_results[session_name].append(get_session(table, model_group), ignore_index=True)
                if specs_name in csv: 
                    table_results[specs_name] = table_results[specs_name].append(get_specs(table, args.branch, args.commit_hash, date_time), ignore_index=True)
                if fail_name in csv:
                    table_results[fail_name] = table_results[fail_name].append(get_failures(table, model_group), ignore_index=True)
                if latency_name in csv:
                    table_results[memory_name] = table_results[memory_name].append(get_memory(table, model_group), ignore_index=True)
                    table_results[latency_name] = table_results[latency_name].append(get_latency(table, model_group), ignore_index=True)
                    table_results[latency_over_time_name] = table_results[latency_over_time_name].append(get_latency_over_time(args.commit_hash, args.report_url, args.branch, table_results[latency_name]), ignore_index=True)
                if status_name in csv: 
                    table_results[status_name] = table_results[status_name].append(get_status(table, model_group), ignore_index=True)
            os.chdir(result_file)
        for table in tables: 
            print('writing ' + table + ' to database')
            db_table_name = 'ep_model_' + table
            write_table(ingest_client, table_results[table], db_table_name, date_time, identifier)

    except BaseException as e: 
        print(str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
