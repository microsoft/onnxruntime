import pandas as pd
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
    common = b.merge(a, on=['model','ep','error type','error message'])
    diff = b.append(common, ignore_index=True).drop_duplicates(['model', 'ep', 'error type', 'error message'], keep=False).loc[:b.index.max()]
    diff.to_csv(args.output_csv)

if __name__=='__main__':
    main()
