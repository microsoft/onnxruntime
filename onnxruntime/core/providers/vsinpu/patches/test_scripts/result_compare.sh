#!/bin/bash
res_file_dir=$1
output_num=$2

# specifying N value
N=5

for i in $(seq 0 $((output_num-1)));
do
  # 构建文件名
  golden_file="${res_file_dir}/expected_res${i}.txt"
  npu_file="${res_file_dir}/npu_res${i}.txt"

  echo "Comparing Top-${N} for the output_${i}"
  python3 compare_topn.py $N $golden_file $npu_file

  echo "--------------------------------"

  echo "Comparing Cosine Similarity for output_${i}:"
  python3 compare_cosine_sim.py $golden_file $npu_file

  echo ""
done
