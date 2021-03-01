# https://docs.microsoft.com/en-us/azure/mysql/connect-python

import mysql.connector
from mysql.connector import errorcode
import git
import os

import argparse
from datetime import datetime

def get_repo_commit(repo_path):
    repo = git.Repo(repo_path, search_parent_directories=True)
    sha = repo.head.object.hexsha
    short_sha = repo.git.rev_parse(sha, short=4)
    return short_sha

create_table_script = "CREATE TABLE perf_test_training_ort_module_data (\
    id int(11) NOT NULL AUTO_INCREMENT,\
    Model varchar(64) COLLATE utf8_bin DEFAULT NULL,\
    BatchId varchar(32) COLLATE utf8_bin DEFAULT NULL,\
    CommitId varchar(32) COLLATE utf8_bin DEFAULT NULL,\
    ModelName varchar(256) COLLATE utf8_bin DEFAULT NULL,\
    DisplayName varchar(512) COLLATE utf8_bin DEFAULT NULL,\
    UseMixedPrecision tinyint(1) DEFAULT NULL,\
    UseAutoCast tinyint(1) DEFAULT NULL,\
    UseDeepSpeed tinyint(1) DEFAULT NULL,\
    Optimizer varchar(32) COLLATE utf8_bin DEFAULT NULL,\
    BatchSize int(11) DEFAULT NULL,\
    SeqLen int(11) DEFAULT NULL,\
    PredictionsPerSeq int(11) DEFAULT NULL,\
    NumOfBatches int(11) DEFAULT NULL,\
    WeightUpdateSteps int(11) DEFAULT NULL,\
    Round int(11) DEFAULT NULL,\
    GradAccSteps int(11) DEFAULT NULL,\
    AvgTimePerBatch float DEFAULT NULL,\
    Throughput float DEFAULT NULL,\
    StabilizedThroughput float DEFAULT NULL,\
    EndToEndThroughput float DEFAULT NULL,\
    TotalTime float DEFAULT NULL,\
    AvgCPU int(11) DEFAULT NULL,\
    Memory int(11) DEFAULT NULL,\
    RunConfig varchar(2048) COLLATE utf8_bin DEFAULT NULL,\
    Time datetime DEFAULT NULL,\
    PRIMARY KEY (id),\
    UNIQUE KEY config_unique (Model,BatchId,CommitId,UseMixedPrecision,UseAutoCast,UseDeepSpeed,Optimizer,BatchSize,SeqLen,ModelName)\
) ENGINE=InnoDB AUTO_INCREMENT=1696 DEFAULT CHARSET=utf8 COLLATE=utf8_bin;"

insert_table_script = "INSERT INTO onnxruntime.perf_test_training_ort_module_data\
    (\
    Model,\
    BatchId,\
    CommitId,\
    ModelName,\
    DisplayName,\
    UseMixedPrecision,\
    UseAutoCast,\
    UseDeepSpeed,\
    Optimizer,\
    BatchSize,\
    SeqLen,\
    PredictionsPerSeq,\
    NumOfBatches,\
    WeightUpdateSteps,\
    Round,\
    GradAccSteps,\
    AvgTimePerBatch,\
    Throughput,\
    StabilizedThroughput,\
    EndToEndThroughput,\
    TotalTime,\
    AvgCPU,\
    Memory,\
    RunConfig,\
    Time)\
VALUES\
(\
\"{Model}\",\
\"{BatchId}\",\
\"{CommitId}\",\
\"{ModelName}\",\
\"{DisplayName}\",\
{UseMixedPrecision},\
{UseAutoCast},\
{UseDeepSpeed},\
\"{Optimizer}\",\
{BatchSize},\
{SeqLen},\
{PredictionsPerSeq},\
{NumOfBatches},\
{WeightUpdateSteps},\
{Round},\
{GradAccSteps},\
{AvgTimePerBatch},\
{Throughput},\
{StabilizedThroughput},\
{EndToEndThroughput},\
{TotalTime},\
{AvgCPU},\
{Memory},\
\"{RunConfig}\",\
\"{Time}\")";

# Obtain connection string information from the portal
def ConnectToPerfDashboardDb(mysql_server_name, power_bi_user_name, password, database):
    config = {
        'host': mysql_server_name,
        'user': power_bi_user_name,
        'password': password,
        'database': database,
    }

    try:
        conn = mysql.connector.connect(**config)
        print("Connection established")
        return conn
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with the user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)

def log_perf_metrics(perf_metrics,
    mysql_server_name, power_bi_user_name, power_bi_password, power_bi_database, perf_repo_path=None):
    if perf_repo_path:
        perf_metrics['CommitId'] = get_repo_commit(perf_repo_path)
    else:
        perf_metrics['CommitId'] = get_repo_commit(os.path.realpath(__file__))

    ConnectAndInsertPerfMetrics(
        mysql_server_name,
        power_bi_user_name,
        power_bi_password,
        power_bi_database,
        perf_metrics)

required_attributes_for_perf_metrics = ['model_name', 'optimizer', 'batch_size', 'epochs', 'train_steps',
    'sequence_length']

def calculate_and_log_perf_metrics(args, start_time,
    mysql_server_name, power_bi_user_name, power_bi_password, power_bi_database, ort_repo_path=None):
    completion_time = datetime.datetime.now()
    perf_metrics_duration = completion_time - start_time

    for attribute in required_attributes_for_perf_metrics:
        if not hasattr(args, attribute):
            raise ValueError('args does not contain all attributes needed to calculate perf metrics. \
                Please prepare perf_metrics and call log_perf_metrics instead')

    perf_metrics = {}    
    perf_metrics['Model'] = args.model_name
    perf_metrics['BatchId'] = 'NA'
    perf_metrics['ModelName'] = args.model_name
    perf_metrics['DisplayName'] = args.model_name
    perf_metrics['UseMixedPrecision'] = args.fp16 if hasattr(args, 'fp16') else False
    perf_metrics['UseAutoCast'] = args.use_auto_cast if hasattr(args, 'use_auto_cast') else False
    perf_metrics['UseDeepSpeed'] = args.use_deep_speed if hasattr(args, 'use_deep_speed') else False
    perf_metrics['Optimizer'] = args.optimizer
    perf_metrics['BatchSize'] = args.batch_size
    perf_metrics['SeqLen'] = args.sequence_length
    perf_metrics['PredictionsPerSeq'] = args.prediction_per_seq if hasattr(args, 'prediction_per_seq') else 0
    perf_metrics['NumOfBatches'] = args.epochs * args.train_steps
    perf_metrics['WeightUpdateSteps'] = args.epochs * args.train_steps
    perf_metrics['Round'] = 0                                       # NA
    perf_metrics['GradAccSteps'] = args.gradient_accumulation_steps

    perf_metrics['AvgTimePerBatch'] = \
        perf_metrics_duration.microseconds / args.train_steps

    perf_metrics['Throughput'] = \
        args.batch_size * args.train_steps / perf_metrics_duration.seconds

    perf_metrics['StabilizedThroughput'] = 0    # TODO
    perf_metrics['EndToEndThroughput'] = 0      # TODO
    perf_metrics['TotalTime'] = perf_metrics_duration.seconds

    perf_metrics['AvgCPU'] = 0                  # TODO
    perf_metrics['Memory'] = 0                  # TODO
    perf_metrics['RunConfig'] = 'na'
    perf_metrics['Time'] = completion_time.strftime("%Y-%m-%d %H:%M:%S")

    log_perf_metrics(perf_metrics, mysql_server_name, power_bi_user_name, power_bi_password, power_bi_database,
        ort_repo_path)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mysql_server_name', help='Perf dashboard mysql server name')
    parser.add_argument('--power_bi_user_name', help='Power BI user name')
    parser.add_argument('--password', help='password', default=None)
    parser.add_argument('--database', help='The dashboard database')
    return parser.parse_args()

def ConnectAndInsertPerfMetrics(mysql_server_name, power_bi_user_name, password, database, perf_metrics):
    conn = ConnectToPerfDashboardDb(mysql_server_name, power_bi_user_name, password, database)
    insert_table_script_values = insert_table_script.format(
        Model=perf_metrics['Model'],
        BatchId=perf_metrics['BatchId'],
        CommitId=perf_metrics['CommitId'],
        ModelName=perf_metrics['ModelName'],
        DisplayName=perf_metrics['DisplayName'],
        UseMixedPrecision=perf_metrics['UseMixedPrecision'],
        UseAutoCast=perf_metrics['UseAutoCast'],
        UseDeepSpeed=perf_metrics['UseDeepSpeed'],
        Optimizer=perf_metrics['Optimizer'],
        BatchSize=perf_metrics['BatchSize'],
        SeqLen=perf_metrics['SeqLen'],
        PredictionsPerSeq=perf_metrics['PredictionsPerSeq'],
        NumOfBatches=perf_metrics['NumOfBatches'],
        WeightUpdateSteps=perf_metrics['WeightUpdateSteps'],
        Round=perf_metrics['Round'],
        GradAccSteps=perf_metrics['GradAccSteps'],
        AvgTimePerBatch=perf_metrics['AvgTimePerBatch'],
        Throughput=perf_metrics['Throughput'], 
        StabilizedThroughput=perf_metrics['StabilizedThroughput'],
        EndToEndThroughput=perf_metrics['EndToEndThroughput'],
        TotalTime=perf_metrics['TotalTime'],
        AvgCPU=perf_metrics['AvgCPU'],
        Memory=perf_metrics['Memory'],
        RunConfig=perf_metrics['RunConfig'],
        Time=perf_metrics['Time'])
    conn.cursor().execute(insert_table_script_values)
    conn.commit()
    conn.cursor().close()
    conn.close()
    print("Done.")

if __name__ == '__main__':
    args = parse_arguments()
    conn = ConnectToPerfDashboardDb(args.mysql_server_name, args.power_bi_user_name, args.password, args.database)
    conn.cursor().execute(create_table_script)
