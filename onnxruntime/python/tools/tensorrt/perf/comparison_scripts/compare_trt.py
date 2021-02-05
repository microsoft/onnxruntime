import pandas as pd
import numpy as np
import argparse

def parse_arguments():  
    # create parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prev", required=True, help="previous csv")
    parser.add_argument("-c", "--current", required=True, help="current csv")
    parser.add_argument("-o", "--output_csv", required=True, help="output different csv")
    args = parser.parse_args()
    return args 

def main():
    args = parse_arguments()
    a = pd.read_csv(args.prev)
    b = pd.read_csv(args.current)
    common = a.merge(b, on=['Model'])
    common['greater'] = np.where((common['Standalone TRT fp32 \nmean (ms)_x'] > common['Standalone TRT fp32 \nmean (ms)_y']) | (common['Standalone TRT fp16 \nmean (ms)_x'] > common['Standalone TRT fp16 \nmean (ms)_y']), True, False)
    greater = common[common['greater'] == True].drop(['greater'], axis=1) 
    greater = greater[greater.keys().sort_values()]
    greater.to_csv(args.output_csv)

if __name__=='__main__':
    main()
