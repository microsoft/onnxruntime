#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


# command line arguments
# --report_url=<string>
# --report_file=<string, local file path>
# --commit_hash=<string, full git commit hash>

import argparse
import mysql.connector
import json
import sys
import os


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="ONNXRuntime test coverge report uploader for dashboard")
    parser.add_argument("--report_url", help="URL to the LLVM json report")
    parser.add_argument(
        "--report_file", help="Path to the local cobertura XML report")
    parser.add_argument("--commit_hash", help="Full Git commit hash")
    return parser.parse_args()


def parse_xml_report(report_file):
    result = {}
    with open(report_file) as json_file:
        data = json.load(json_file)

    linestat = data['data'][0]['totals']['lines']
    result['coverage'] = float(linestat['percent']/100.0)
    result['lines_covered'] = int(linestat['covered'])
    result['lines_valid'] = int(linestat['count'])
    return result


def write_to_db(coverage_data, args):
    # connect to database

    cnx = mysql.connector.connect(
        user='ort@onnxruntimedashboard',
        password=os.environ.get('DASHBOARD_MYSQL_ORT_PASSWORD'),
        host='onnxruntimedashboard.mysql.database.azure.com',
        database='onnxruntime')

    try:
        cursor = cnx.cursor()

        # delete old records
        delete_query = ('DELETE FROM onnxruntime.test_coverage '
                        'WHERE UploadTime < DATE_SUB(Now(), INTERVAL 30 DAY);'
                        )

        cursor.execute(delete_query)

        # insert current record
        insert_query = ('INSERT INTO onnxruntime.test_coverage '
                        '(UploadTime, CommitId, Coverage, LinesCovered, TotalLines, ReportURL) '
                        'VALUES (Now(), "%s", %f, %d, %d, "%s") '
                        'ON DUPLICATE KEY UPDATE '
                        'UploadTime=Now(), Coverage=%f, LinesCovered=%d, TotalLines=%d, ReportURL="%s";'
                        ) % (args.commit_hash,
                             coverage_data['coverage'],
                             coverage_data['lines_covered'],
                             coverage_data['lines_valid'],
                             args.report_url,
                             coverage_data['coverage'],
                             coverage_data['lines_covered'],
                             coverage_data['lines_valid'],
                             args.report_url
                             )

        cursor.execute(insert_query)
        cnx.commit()

        # # Use below for debugging:
        # cursor.execute('select * from onnxruntime.test_coverage')
        # for r in cursor:
        #     print(r)

        cursor.close()
        cnx.close()
    except BaseException as e:
        cnx.close()
        raise e


if __name__ == "__main__":
    try:
        args = parse_arguments()
        coverage_data = parse_xml_report(args.report_file)
        write_to_db(coverage_data, args)
    except BaseException as e:
        print(str(e))
        sys.exit(1)
