
import csv
import datetime
import json
import os
import sys
import time

from tqdm import tqdm
from transformers import modeling_tf_pytorch_utils

from model import ModelImp
from NWModelEval import calculate_appg
from myutils import outFileHandler

def add_time_string_to_file(filename):
    dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    f_basename, f_ext = os.path.split(filename)
    return f"f{f_basename}_{dt}{f_ext}"
def calc_appg(input_file, output_file, refFile, model):
    def format_for_eval(prefix, completions):
        if completions is None or len(completions) <= 0:
            return prefix, "NULL"
        else:
            return prefix, ",".join(completions)
    #output_file = add_time_string_to_file(output_file)
    print(output_file)
    total_time = 0
    count = 0
    sent_queries = []
    with  open(input_file, 'r', encoding = 'utf-8', errors='ignore') as input_fp:
        with open(output_file, "w", encoding = 'utf-8', errors='ignore') as output_fp:
            temporary_file = open(".\\temp_file.csv", 'w+')
            outFileHandler(temporary_file)

            for line in tqdm(input_fp):
            # line = input_fp.readline()
            # while(line):
                start = time.time()
                query_prefix = line.split('\t')[0].strip('\n')
                if query_prefix not in sent_queries:
                    sent_queries.append(query_prefix)
                    result = json.loads(model.Eval(query_prefix))
                    total_time += time.time() - start
                    csv.writer(output_fp, delimiter="\t", lineterminator="\n").writerow(format_for_eval(query_prefix, result))
                    count += 1
                # line = input_fp.readline()
            temporary_file.close()
    print("average latency: ", total_time * 1000. / count)
    return calculate_appg(refFile, output_file, False)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(1)
    model = ModelImp()
    calc_appg(sys.argv[1], sys.argv[2], sys.argv[3], model)

