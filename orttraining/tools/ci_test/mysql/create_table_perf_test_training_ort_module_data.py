# https://docs.microsoft.com/en-us/azure/mysql/connect-python

import mysql.connector
from mysql.connector import errorcode

import argparse
from datetime import datetime

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

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', help='what to do')
    parser.add_argument('--mysql_server_name', help='Perf dashboard mysql server name')
    parser.add_argument('--power_bi_user_name', help='Power BI user name')
    parser.add_argument('--password', help='password', default=None)
    parser.add_argument('--database', help='The dashboard database')
    parser.add_argument('--perf_metrics', help='The perf metrics')
    return parser.parse_args()

def ConnectAndInsertPerfMatrics(mysql_server_name, power_bi_user_name, password, database, perf_metrics):
    conn = ConnectToPerfDashboardDb(args.mysql_server_name, args.power_bi_user_name, args.password, args.database)
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
    if args.action == 'create_perf_table':
        conn = ConnectToPerfDashboardDb(args.mysql_server_name, args.power_bi_user_name, args.password, args.database)
        conn.cursor().execute(create_table_script)
    elif args.action == 'insert_perf_table':
        perf_metrics = {}
        perf_metrics['Model'] = "model"
        perf_metrics['BatchId'] = "batch_id2"
        perf_metrics['CommitId'] = "commit id"
        perf_metrics['ModelName'] = "model name"
        perf_metrics['DisplayName'] = "disp name"
        perf_metrics['UseMixedPrecision'] = True
        perf_metrics['UseAutoCast'] = False
        perf_metrics['UseDeepSpeed'] = True
        perf_metrics['Optimizer'] = "optim"
        perf_metrics['BatchSize'] = 20
        perf_metrics['SeqLen'] = 128
        perf_metrics['PredictionsPerSeq'] = 200
        perf_metrics['NumOfBatches'] = 300
        perf_metrics['WeightUpdateSteps'] = 20
        perf_metrics['Round'] = 10
        perf_metrics['GradAccSteps'] = 12
        perf_metrics['AvgTimePerBatch'] = 0.2
        perf_metrics['Throughput'] = 20
        perf_metrics['StabilizedThroughput'] = 30
        perf_metrics['EndToEndThroughput'] = 40
        perf_metrics['TotalTime'] = 2.3
        perf_metrics['AvgCPU'] = 2
        perf_metrics['Memory'] = 3
        perf_metrics['RunConfig'] ="run  config"
        perf_metrics['Time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        ConnectAndInsertPerfMatrics(args.mysql_server_name, args.power_bi_user_name, args.password, args.database, perf_metrics)
