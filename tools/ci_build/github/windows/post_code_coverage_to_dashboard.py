#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


# command line arguments
# --report_url=<string>
# --report_file=<string, local file path, TXT/JSON file>
# --commit_hash=<string, full git commit hash>
# --build_config=<string, JSON format specifying os, arch and config>

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
        "--report_file", help="Path to the local JSON/TXT report", required=True)
    parser.add_argument("--commit_hash", help="Full Git commit hash", required=True)
    parser.add_argument("--build_config", help="Build configuration, os, arch and config, in JSON format")
    return parser.parse_args()


def parse_txt_report(report_file):
    data = {}
    with open(report_file, 'r') as report:
        for line in reversed(report.readlines()):
            if 'TOTAL' in line:
                fields = line.strip().split()
                data['lines_valid'] = int(fields[1])
                data['lines_covered'] = int(fields[2])
                data['coverage'] = float(fields[3].strip('%'))/100
                break
    return data


def parse_json_report(report_file):
    result = {}
    with open(report_file) as json_file:
        data = json.load(json_file)

    linestat = data['data'][0]['totals']['lines']
    result['coverage'] = float(linestat['percent']/100.0)
    result['lines_covered'] = int(linestat['covered'])
    result['lines_valid'] = int(linestat['count'])
    return result


def write_to_db(coverage_data, build_config, args):
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
                        '''(UploadTime, CommitId, Coverage, LinesCovered, TotalLines, OS,
                          Arch, BuildConfig, ReportURL) '''
                        'VALUES (Now(), "%s", %f, %d, %d, "%s", "%s", "%s", "%s") '
                        'ON DUPLICATE KEY UPDATE '
                        '''UploadTime=Now(), Coverage=%f, LinesCovered=%d, TotalLines=%d,
                          OS="%s", Arch="%s", BuildConfig="%s", ReportURL="%s"; '''
                        ) % (args.commit_hash,
                             coverage_data['coverage'],
                             coverage_data['lines_covered'],
                             coverage_data['lines_valid'],
                             build_config.get('os', 'win'),
                             build_config.get('arch', 'x64'),
                             build_config.get('config', 'default'),
                             args.report_url,
                             coverage_data['coverage'],
                             coverage_data['lines_covered'],
                             coverage_data['lines_valid'],
                             build_config.get('os', 'win'),
                             build_config.get('arch', 'x64'),
                             build_config.get('config', 'default'),
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
        if args.report_file.endswith(".json"):
            coverage_data = parse_json_report(args.report_file)
        elif args.report_file.endswith(".txt"):
            coverage_data = parse_txt_report(args.report_file)
        else:
            raise ValueError("Only report extensions txt or json are accepted")

        build_config = json.loads(args.build_config) if args.build_config else {}
        write_to_db(coverage_data, build_config, args)
    except BaseException as e:
        print(str(e))
        sys.exit(1)
