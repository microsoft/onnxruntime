#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import mysql.connector
import sys
import os
import subprocess
import pandas as pd
from sqlalchemy import create_engine

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", "--report_folder", help="Path to the local file report", required=True)
    return parser.parse_args()

def parse_csv(report_file):
    table = pd.read_csv(report_file)
    return table

def adjust_columns(table, columns, db_columns, model_group): 
    table = table[columns]
    table = table.set_axis(db_columns, axis=1)
    table = table.assign(Group=model_group)
    return table 

def get_failures(fail, model_group):
    fail_columns = fail.keys()
    fail_db_columns = ['Model', 'Ep', 'ErrorType', 'ErrorMessage']
    fail = adjust_columns(fail, fail_columns, fail_db_columns, model_group)
    return fail

def get_memory(memory, model_group): 
    memory_columns = ['Model', \
                      'CUDA EP fp32 \nmemory usage (MiB)', \
                      'TRT EP fp32 \nmemory usage (MiB)', \
                      'Standalone TRT fp32 \nmemory usage (MiB)', \
                      'CUDA EP fp16 \nmemory usage (MiB)', \
                      'TRT EP fp16 \nmemory usage (MiB)', \
                      'Standalone TRT fp16 \nmemory usage (MiB)' \
                      ]
    memory_db_columns = ['Model', 'CudaFp32', 'TrtFp32', 'StandaloneFp32', 'CudaFp16', 'TrtFp16', 'StandaloneFp16']
    memory = adjust_columns(memory, memory_columns, memory_db_columns, model_group)
    return memory
                    
def get_latency_fp32(latency, model_group):
    latency_fp32_columns = ['Model', \
                            'CPU \nmean (ms)', \
                            'CUDA fp32 \nmean (ms)', \
                            'TRT EP fp32 \nmean (ms)', \
                            'Standalone TRT fp32 \nmean (ms)', \
                            'TRT v CUDA EP fp32 \ngain (mean) (%)', \
                            'EP v Native TRT fp32 \ngain (mean) (%)' \
                            ]    
    latency_db_columns = ['Model', 'Cpu', 'CudaEp', 'TrtEp', 'Standalone', 'CudaTrtGain', 'NativeEpGain']
    latency = adjust_columns(latency, latency_fp32_columns, latency_db_columns, model_group)
    return latency

def get_latency_fp16(latency, model_group):
    latency_fp16_columns = ['Model', \
                            'CPU \nmean (ms)', \
                            'CUDA fp16 \nmean (ms)', \
                            'TRT EP fp16 \nmean (ms)', \
                            'Standalone TRT fp16 \nmean (ms)', \
                            'TRT v CUDA EP fp16 \ngain (mean) (%)', \
                            'EP v Native TRT fp16 \ngain (mean) (%)' \
                            ]    
    latency_db_columns = ['Model', 'Cpu', 'CudaEp', 'TrtEp', 'Standalone', 'CudaTrtGain', 'NativeEpGain']
    latency = adjust_columns(latency, latency_fp16_columns, latency_db_columns, model_group)
    return latency
    
def get_status(status, model_group):
    status_columns = status.keys()
    status_db_columns = ['Model', 'Cpu', 'CudaEpFp32', 'TrtEpFp32', 'StandaloneFp32', 'CudaEpFp16', 'TrtEpFp16', 'StandaloneFp16']
    status = adjust_columns(status, status_columns, status_db_columns, model_group)
    return status

def get_database_cert(): 
    cert = 'BaltimoreCyberTrustRoot.crt.pem'
    if not os.path.exists(cert):
        p = subprocess.run(["wget", "https://cacerts.digicert.com/DigiCertGlobalRootG2.crt.pem", "-O", cert], check=True)
    return cert 

def write_table(engine, table, table_name): 
    table.to_sql(table_name, con=engine, if_exists='replace', index=False, chunksize=1)

def main():
    
    # connect to database 
    cert = get_database_cert()
    ssl_args = {'ssl_ca': cert}
    connection_string = 'mysql+mysqlconnector://' + \
                        'powerbi@onnxruntimedashboard:' + \
                        os.environ.get('DASHBOARD_MYSQL_ORT_PASSWORD') + \
                        '@onnxruntimedashboard.mysql.database.azure.com/' + \
                        'onnxruntime'
    engine = create_engine(connection_string, connect_args=ssl_args)

    try: 
        args = parse_arguments()
        result_file = args.report_folder

        folders = os.listdir(result_file)
        os.chdir(result_file)
       
        
        fail = pd.DataFrame()
        memory = pd.DataFrame()
        latency_fp32 = pd.DataFrame()
        latency_fp16 = pd.DataFrame()
        status = pd.DataFrame()
       
        for model_group in folders: 
            os.chdir(model_group)
            csv_filenames = os.listdir()
            for csv in csv_filenames: 
                table = parse_csv(csv)
                if "fail" in csv:
                    fail = fail.append(get_failures(table, model_group), ignore_index=True)
                if "latency" in csv:
                    memory = memory.append(get_memory(table, model_group), ignore_index=True)
                    latency_fp32 = latency_fp32.append(get_latency_fp32(table, model_group), ignore_index=True)
                    latency_fp16 = latency_fp16.append(get_latency_fp16(table, model_group), ignore_index=True)
                if "status" in csv: 
                    status = status.append(get_status(table, model_group), ignore_index=True)
            os.chdir(result_file)
    
        write_table(engine, fail, 'ep_model_fails')
        write_table(engine, memory, 'ep_model_memory')
        write_table(engine, latency_fp32, 'ep_model_latency_fp32')
        write_table(engine, latency_fp16, 'ep_model_latency_fp16')
        write_table(engine, status, 'ep_models_status')

    except BaseException as e: 
        print(str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()






