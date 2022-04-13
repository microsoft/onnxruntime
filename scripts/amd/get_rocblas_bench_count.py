import pandas as pd
import argparse
import re
import os


def get_rocblas_bench_count(bench_log_path):
    rocblas_bench_commands=[]
    with open(bench_log_path) as f:
        for line in f:
            search_results = re.search("(./rocblas-bench.*)",line)
            if search_results:
                rocblas_bench_commands.append(line.strip())
            
    df=pd.DataFrame(rocblas_bench_commands)   
    df=df.iloc[:,0].value_counts().to_frame().reset_index()
    df.columns=["rocblas-bench command","count"]
    df=df[["count","rocblas-bench command"]]
    df.to_csv(os.path.splitext(bench_log_path)[0]+"_count.csv",index=False)
    
    return df

parser = argparse.ArgumentParser()
parser.add_argument('rocblas_log_bench', type=str, help='bert training directory')
args = parser.parse_args()

if args.rocblas_log_bench:
    get_rocblas_bench_count(args.rocblas_log_bench)
