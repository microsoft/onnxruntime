import re
import argparse
import pprint

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str)

regexp="run_res_accu-[0-9]+_maxseq-([0-9]+)_([0-9a-zA-Z_-]+)(fp[0-9]+_[a-z0-9]+_[a-z0-9]+):Throughput: ([0-9]+.[0-9]+) Examples \/ Second"

args = parser.parse_args()
fileName=args.path

sums = {}
freqs = {}
with open(fileName) as f:
  for line in f:
    match = re.match(regexp, line)
    if match:
      key = str(match.group(1)) + match.group(3)
      throuput = match.group(4)
      if key not in sums:
        sums[key] = 0.0
        freqs[key] = 0

      sums[key] = sums[key] + float(throuput) 
      freqs[key] += 1
    else:
      raise Exception("warning: the line is not parsed correctly ", line)


print("('fptype_gpunum_batch', throughput , collected_run_cnt)")
for k in sorted(sums):
  print(k, sums[k], freqs[k])
