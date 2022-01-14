import argparse
import sys
import os
import pandas as pd
import time
from datetime import datetime, timedelta
from azure.kusto.data import KustoConnectionStringBuilder
from azure.kusto.data.helpers import dataframe_from_result_table 
from azure.kusto.ingest import (
    IngestionProperties,
    DataFormat,
    ReportLevel,
    QueuedIngestClient,
)
from perf_utils import *

# database connection strings 
cluster_ingest = "https://ingest-onnxruntimedashboarddb.southcentralus.kusto.windows.net"
database = "ep_perf_dashboard"

# table names
fail = 'fail'
memory = 'memory'
latency = 'latency'
status = 'status'
latency_over_time = 'latency_over_time'
specs = 'specs' 

time_string_format = '%Y-%m-%d %H:%M:%S'

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

def get_specs(specs, branch, commit_id):
    specs = specs.append({'.': 6, 'Spec': 'Branch', 'Version' : branch}, ignore_index=True)
    specs = specs.append({'.': 7, 'Spec': 'CommitId', 'Version' : commit_id}, ignore_index=True)
    return specs

def write_table(ingest_client, table, table_name, trt_version, upload_time):
    if table.empty:
        return
    table = table.assign(TrtVersion=trt_version) # add TrtVersion
    table = table.assign(UploadTime=upload_time) # add UploadTime
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

def main():
    
    args = parse_arguments()
    
    # connect to database
    kcsb_ingest = KustoConnectionStringBuilder.with_az_cli_authentication(cluster_ingest)
    ingest_client = QueuedIngestClient(kcsb_ingest)
    date_time = get_time()

    try:
        result_file = args.report_folder

        folders = os.listdir(result_file)
        os.chdir(result_file)

        tables = [fail, memory, latency, status, latency_over_time, specs]
        table_results = {}
        for table_name in tables:
            table_results[table_name] = pd.DataFrame()

        for model_group in folders:
            os.chdir(model_group)
            csv_filenames = os.listdir()
            for csv in csv_filenames:
                table = parse_csv(csv)
                if specs in csv: 
                    table_results[specs] = table_results[specs].append(get_specs(table, args.branch, args.commit_hash), ignore_index=True)
                if fail in csv:
                    table_results[fail] = table_results[fail].append(get_failures(table, model_group), ignore_index=True)
                if latency in csv:
                    table_results[memory] = table_results[memory].append(get_memory(table, model_group), ignore_index=True)
                    table_results[latency] = table_results[latency].append(get_latency(table, model_group), ignore_index=True)
                    table_results[latency_over_time] = table_results[latency_over_time].append(get_latency_over_time(args.commit_hash, args.report_url, args.branch, table_results[latency]), ignore_index=True)
                if status in csv: 
                    table_results[status] = table_results[status].append(get_status(table, model_group), ignore_index=True)
            os.chdir(result_file)
        for table in tables: 
            print('writing ' + table + ' to database')
            db_table_name = 'ep_model_' + table
            write_table(ingest_client, table_results[table], db_table_name, args.trt_version, date_time)

    except BaseException as e: 
        print(str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
