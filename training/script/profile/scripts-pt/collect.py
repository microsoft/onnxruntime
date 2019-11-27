import re
import argparse
import pprint

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str)

#2019-11-16T18:08:06.000Z /container_e21_1570560736018_2773_01_001716: fp16_g16_phase1b64_phase2b14: training throughput phase1: 2877.44 sequences/second
regexp="([0-9.a-zA-Z_\-:\/ ]*)(fp[0-9]+)_g([0-9]+)_phase1b([0-9]+)_phase2b([0-9]+): training throughput phase([0-9]+): ([0-9]*.*[0-9]*) sequences\/second"

args = parser.parse_args()
fileName=args.path

sums={}
freqs={}
with open(fileName) as f:
  for line in f:
    match = re.match(regexp, line)
    if match:
      if match.group(6) == "1":
        batch=match.group(4)
      else:
        batch=match.group(5)
      key=match.group(2) + "_g" + str(match.group(3)).zfill(2) + "_phase" + match.group(6) + "_b" + str(batch).zfill(3)

      throughput = match.group(7)
      if throughput == "":
          throughput = "-1" # negative number means no throughput returned
      if key not in sums:
        sums[key] = 0.0
        freqs[key] = 0
      sums[key] = sums[key] + float(throughput)
      freqs[key] += 1
    else:
      raise Exception("warning: the line is not parsed correctly ", line)

res={}
for r in sums:
  if r in freqs and freqs[r] != 0:
    res[r] = sums[r] / float(freqs[r])

print("('fptype_gpu_phase_batch', averaged throughput , collected_run_cnt)")
for k in sorted(res):
  print(k, res[k], freqs[k])
