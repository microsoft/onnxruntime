# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""
Parse and Post customized perf data to DB
"""

import argparse
import csv
import datetime
import logging
import os
import pandas as pd
import sys
from azure.kusto.data import KustoConnectionStringBuilder
from azure.kusto.data.data_format import DataFormat
from azure.kusto.ingest import IngestionProperties, QueuedIngestClient, ReportLevel

# Constants, replace with actual values
MODEL = "llama-2.0"
DEVICE_ID = "Samsung Galaxy S24"

# Configure logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

def parse_mobile_perf(log_data: str, csv_filename: str):
    """
    Parse log data and save metrics to a CSV file.

    Args:
        log_data (str): The log data to parse.
        csv_filename (str): The filename to save the parsed metrics.
    """
    metrics = {
        "Model": MODEL,
        "DeviceId": DEVICE_ID,
        "TTFTAvgTimeSec": None,
        "TTFTAvgTokenPerSec": None,
        "TokenGenerationAvgTimeSec": None,
        "TokenGenerationAvgTokenPerSec": None,
        "TokenSamplingAvgTimeSec": None,
        "TokenSamplingAvgTokenPerSec": None,
        "E2EGenerationAvgTimeSec": None,
        "PeakMemoryMB": None
    }

    current_section = None

    for line in log_data.split("\n"):
        line = line.strip()
    
        if "Prompt processing" in line:
            current_section = "TTFT"
        elif "Token generation" in line:
            current_section = "TokenGeneration"
        elif "Token sampling" in line:
            current_section = "TokenSampling"
        elif "E2E generation" in line:
            current_section = "E2EGeneration"
        elif "Peak working set size" in line:
            current_section = "PeakMemory"

        if line.startswith("avg (us):"):
            value = float(line.split(":")[1].strip()) / 1_000_000
            if current_section == "TTFT":
                metrics["TTFTAvgTimeSec"] = value
            elif current_section == "TokenGeneration":
                metrics["TokenGenerationAvgTimeSec"] = value
            elif current_section == "TokenSampling":
                metrics["TokenSamplingAvgTimeSec"] = value

        elif line.startswith("avg (tokens/s):"):
            value = float(line.split(":")[1].strip())
            if current_section == "TTFT":
                metrics["TTFTAvgTokenPerSec"] = value
            elif current_section == "TokenGeneration":
                metrics["TokenGenerationAvgTokenPerSec"] = value
            elif current_section == "TokenSampling":
                metrics["TokenSamplingAvgTokenPerSec"] = value

        elif line.startswith("avg (ms):"):
            value = float(line.split(":")[1].strip()) / 1_000
            if current_section == "E2EGeneration":
                metrics["E2EGenerationAvgTimeSec"] = value

        elif line.startswith("Peak working set size (bytes):"):
            value = float(line.split(":")[1].strip()) / (1024 * 1024)
            metrics["PeakMemoryMB"] = value

    for key in metrics:
        if metrics[key] is not None and isinstance(metrics[key], (int, float)):
            metrics[key] = f"{metrics[key]:.8f}"

    save_to_csv(metrics, csv_filename)
    
def save_to_csv(metrics: dict, csv_filename: str):
    try:
        with open(csv_filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(metrics.keys())
            writer.writerow(metrics.values())
        logging.info(f"Metrics saved to {csv_filename}")
    except IOError as e:
        logging.error(f"Failed to save metrics to {csv_filename}: {e}")

def post_to_db(csv_file: str, kusto_table: str, kusto_conn: str, kusto_db: str):
    """
    Post data to Kusto DB.

    Args:
        csv_file (str): The path to csv file.
        kusto_table (str): The Kusto table name.
        kusto_conn (str): The Kusto connection string.
        kusto_db (str): The Kusto database name.
    """
    try:
        table = pd.read_csv(csv_file)
        upload_time = datetime.datetime.now(tz=datetime.timezone.utc).replace(microsecond=0).isoformat()
        table["UploadTime"] = upload_time
        
        kcsb_ingest = KustoConnectionStringBuilder.with_az_cli_authentication(kusto_conn)
        ingest_client = QueuedIngestClient(kcsb_ingest)

        ingestion_props = IngestionProperties(
            database=kusto_db,
            table=kusto_table,
            data_format=DataFormat.CSV,
            report_level=ReportLevel.FailuresAndSuccesses,
        )

        ingest_client.ingest_from_dataframe(table, ingestion_properties=ingestion_props)
        logging.info(f"Data uploaded to Kusto table {kusto_table} in database {kusto_db}")
    except Exception as e:
        logging.error(f"Failed to upload data to Kusto DB: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse and post perf data to DB')
    parser.add_argument('--kusto-table', required=True, help='Kusto table name')
    parser.add_argument('--kusto-conn', required=True, help='Kusto connection string')
    parser.add_argument('--kusto-db', required=True, help='Kusto database name')
    parser.add_argument('--upload-csv', help='CSV file to upload to DB')
    parser.add_argument('--output-csv', default='data.csv', help='CSV file to save parsed metrics')
    # Mobile perf data
    parser.add_argument('--parse-mobile-perf', action='store_true', 
                        help='Parse mobile perf data and post to DB')
    parser.add_argument('--log-file', help='Path to log file containing performance data')
    args = parser.parse_args()

    if args.parse_mobile_perf:
        log_data = """
        Batch size: 1, prompt tokens: 2, tokens to generate: 128
        Prompt processing (time to first token):
               avg (us):       140607
               avg (tokens/s): 14.224
               p50 (us):       141764
               stddev (us):    7045.29
               n:              5 * 2 token(s)
        Token generation:
               avg (us):       185163
               avg (tokens/s): 5.40064
               p50 (us):       191207
               stddev (us):    40530.6
               n:              635 * 1 token(s)
        Token sampling:
               avg (us):       70.8335
               avg (tokens/s): 14117.6
               p50 (us):       61.146
               stddev (us):    28.0707
               n:              640 * 1 token(s)
        E2E generation (entire generation loop):
               avg (ms):       23665.5
               p50 (ms):       23663.7
               stddev (ms):    155.755
               n:              5
        Peak working set size (bytes): 2864254976
        """
        if args.log_file:
            try:
                with open(args.log_file, 'r') as f:
                    log_data = f.read()
                logging.info(f"Read mobile perf log from {args.log_file}")
            except IOError as e:
                logging.error(f"Failed to read log file {args.log_file}: {e}")
                sys.exit(1)
        
        # Parse the mobile perf data for further upload
        parse_mobile_perf(log_data, args.output_csv)
        post_to_db(args.output_csv, args.kusto_table, args.kusto_conn, args.kusto_db)
    elif args.upload_csv:
        # Upload an existing CSV to the database
        # Need to make sure schema is correct
        try:
            post_to_db(args.upload_csv, args.kusto_table, args.kusto_conn, args.kusto_db)
        except Exception as e:
            logging.error(f"Failed to upload CSV {args.upload_csv} to DB: {e}")
            sys.exit(1)
    else:
        parser.print_help()
        logging.error("Either --parse-mobile-perf or --upload-csv must be specified")
        sys.exit(1)