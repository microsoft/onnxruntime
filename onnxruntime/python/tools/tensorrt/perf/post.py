#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import mysql.connector
import sys
import os
import subprocess
from sqlalchemy import create_engine

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", "--report_folder", help="Path to the local file report", required=True)
    return parser.parse_args()

def parse_csv(report_file):
    import pandas as pd
    table = pd.read_csv(report_file)
    return table

def post_latency_fp32(engine, latency, model_group):
    latency_fp32_columns = ['Model', \
                            'CPU \nmean (ms)', \
                            'CUDA fp32 \nmean (ms)', \
                            'TRT EP fp32 \nmean (ms)', \
                            'Standalone TRT fp32 \nmean (ms)', \
                            'TRT v CUDA EP fp32 \ngain (mean) (%)', \
                            'EP v Native TRT fp32 \ngain (mean) (%)' \
                            ]    
    latency = latency[latency_fp32_columns]
    latency_db_columns = ['Model', 'Cpu', 'CudaEp', 'TrtEp', 'Standalone', 'CudaTrtGain', 'NativeEpGain']
    latency = latency.set_axis(latency_db_columns, axis=1)
    latency = latency.assign(Group=model_group)
    latency.to_sql('ep_model_latency_fp32', con=engine, if_exists='replace', index=False, chunksize=1)

def post_latency_fp16(engine, latency, model_group):
    latency_fp16_columns = ['Model', \
                            'CPU \nmean (ms)', \
                            'CUDA fp16 \nmean (ms)', \
                            'TRT EP fp16 \nmean (ms)', \
                            'Standalone TRT fp16 \nmean (ms)', \
                            'TRT v CUDA EP fp16 \ngain (mean) (%)', \
                            'EP v Native TRT fp16 \ngain (mean) (%)' \
                            ]    
    latency = latency[latency_fp16_columns]
    latency_db_columns = ['Model', 'Cpu', 'CudaEp', 'TrtEp', 'Standalone', 'CudaTrtGain', 'NativeEpGain']
    latency = latency.set_axis(latency_db_columns, axis=1)
    latency = latency.assign(Group=model_group)
    latency.to_sql('ep_model_latency_fp16', con=engine, if_exists='replace', index=False, chunksize=1)

def post_status(engine, status, model_group):
    status_db_columns = ['Model', 'Cpu', 'CudaEpFp32', 'TrtEpFp32', 'StandaloneFp32', 'CudaEpFp16', 'TrtEpFp16', 'StandaloneFp16']
    status = status.set_axis(status_db_columns, axis=1)
    status = status.assign(Group=model_group)
    status.to_sql('ep_models_status', con=engine, if_exists='replace', index=False, chunksize=1)

def get_database_cert(): 
    cert = 'BaltimoreCyberTrustRoot.crt.pem'
    if not os.path.exists(cert):
        p = subprocess.run(["wget", "https://cacerts.digicert.com/DigiCertGlobalRootG2.crt.pem", "-O", cert], check=True)
    return cert 

def main():
    
    # connect to database 
    cert = get_database_cert()
    #cert = 'C:\ssl\BaltimoreCyberTrustRoot.crt.pem'
    ssl_args = {'ssl_ca': cert}
    connection_string = 'mysql+mysqlconnector://' + \
                        'powerbi@onnxruntimedashboard:' + \
                        os.environ.get('DASHBOARD_MYSQL_ORT_PASSWORD') + \
                        '@onnxruntimedashboard.mysql.database.azure.com/' + \
                        'onnxruntime'
    print(connection_string)
    engine = create_engine(connection_string, connect_args=ssl_args)

    try: 
        args = parse_arguments()
        result_file = args.report_folder

        folders = os.listdir(result_file)
        os.chdir(result_file)
        for model_group in folders: 
            os.chdir(model_group)
            csv_filenames = os.listdir()
            for csv in csv_filenames: 
                table = parse_csv(csv)
                if "latency" in csv:
                    post_latency_fp32(engine, table, model_group)
                    post_latency_fp16(engine, table, model_group)
                if "status" in csv: 
                    post_status(engine, table, model_group)
            os.chdir(result_file)

    except BaseException as e: 
        print(str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
