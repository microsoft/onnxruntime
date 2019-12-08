import re
import argparse
import pprint

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str)

regexp="run_res_accu-[0-9]+_maxseq-([0-9]+)_([0-9a-zA-Z_-]+)_(fp[0-9]+)_g([0-9]+)_b([0-9]+):Throughput: ([0-9]+.[0-9]+) Examples \/ Second"

args = parser.parse_args()
fileName=args.path

sums = {}
freqs = {}
with open(fileName) as f:
  for line in f:
    match = re.match(regexp, line)
    if match:
      seq_length=str(match.group(1))
      if seq_length == "128":
        phase = 1
      else:
        phase = 2

      key=match.group(3) + "_g" + str(match.group(4)).zfill(2) + "_phase" + str(phase) + "_b" + str(match.group(5)).zfill(3)
      throughput = match.group(6)
      if key not in sums:
        sums[key] = 0.0
        freqs[key] = 0

      sums[key] = sums[key] + float(throughput) 
      freqs[key] += 1
    else:
      raise Exception("warning: the line is not parsed correctly ", line)

print("('fptype_gpu_phase_batch', throughput , collected_run_cnt)")
for k in sorted(sums):
  print(k, sums[k], freqs[k])
