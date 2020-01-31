#!/bin/bash
batch=$1
fpkind=$2
gpu_num=$3
accu_steps=$4
max_seq_length=$5
max_predictions_per_seq=$6
eval_steps=$7
custom_params_string=$CUSTOM_PARAM_STRING
result_path=$RESULT_PATH

printf -v gpu_num_formatted "%02d" $gpu_num
printf -v batch_formatted "%02d" $batch

mkdir -p "/tmp/results"
tmpf="/tmp/results/run_res_accu-"$accu_steps"_maxseq-"$max_seq_length"_local-"$OMPI_COMM_WORLD_LOCAL_RANK"_world-"$OMPI_COMM_WORLD_RANK"_"`uname -n`"_"$fpkind"_g"$gpu_num_formatted"_b"$batch_formatted

DATADIR="/bert_data/"$max_seq_length"/books_wiki_en_corpus"

steps=$eval_steps

declare -a shared_params=( \
    --model_name="/code/bert" \
    --num_train_steps=$steps \
    --train_batch_size=$batch \
    --mode=train \
    --train_data_dir=$DATADIR"/train" \
    --test_data_dir=$DATADIR"/test" \
    --gradient_accumulation_steps=${accu_steps} \
    --optimizer=Lamb \
    --learning_rate=3e-3 \
    --max_seq_length=$max_seq_length \
    --max_predictions_per_seq=$max_predictions_per_seq \
    --warmup_ratio=0.2843 \
    --warmup_mode=Poly \
    --display_loss_steps=100 \
    --log_dir=$PHILLY_JOB_DIRECTORY \
 )

if [ "$fpkind" == "fp16" ]
then
    fp16_params=(\
        --use_mixed_precision=True \
        --allreduce_in_fp16 \
    )
    full_params=( "${shared_params[@]}" "${fp16_params[@]}" "${custom_params_string}")

else
    fp32_params=(\
        --use_mixed_precision=False \
    )
    full_params=( "${shared_params[@]}" "${fp32_params[@]}" "${custom_params_string}")
fi

echo "running command /code/binary/onnxruntime_training_bert "${full_params[@]}

/code/binary/onnxruntime_training_bert ${full_params[@]} >${tmpf} 2>${tmpf}_err
cp $tmpf $result_path"/"
cp $tmpf"_err" $result_path"/"

exit 0