import pandas as pd
import numpy as np
import argparse

ep_map = {"cpu": "CPU", "cuda":"CUDA","trt": "TRT EP","native": "Standalone TRT"}

def parse_arguments():  
    # create parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prev", required=True, help="previous csv")
    parser.add_argument("-c", "--current", required=True, help="current csv")
    parser.add_argument("-o", "--output_csv", required=True, help="output different csv")
    parser.add_argument("--ep", required=False, default="trt", choices=["cpu", "cuda", "trt", "native"], help="ep to capture regressions on")
    parser.add_argument("--tolerance", required=False, default=0, help="allowed tolerance for latency comparison")
    args = parser.parse_args()
    return args 

def get_table_condition(table, fp, ep, tol): 
    ep = ep_map[ep]
    col1 = ep + " " + fp + " \nmean (ms)_x"
    col2 = ep + " " + fp + " \nmean (ms)_y"
    condition = table[col1] > (table[col2] + tol)
    return condition

def main():
    args = parse_arguments()
    a = pd.read_csv(args.prev)
    b = pd.read_csv(args.current)
    
    common = a.merge(b, on=['Model'])
    
    condition_fp32 = get_table_condition(common, "fp32", args.ep, args.tolerance)
    condition_fp16 = get_table_condition(common, "fp16", args.ep, args.tolerance)
    
    common['greater'] = np.where((condition_fp32 | condition_fp16), True, False)
    greater = common[common['greater'] == True].drop(['greater'], axis=1) 
    
    # arrange columns
    keys = list(greater.keys().sort_values())
    keys.insert(0, keys.pop(keys.index('Model')))
    greater = greater[keys]
    
    greater.to_csv(args.output_csv)

if __name__=='__main__':
    main()
