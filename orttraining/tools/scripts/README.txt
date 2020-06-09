Procedure to export NV's pytorch model to ONNX.

1. cd into BERT in DeepLearningExamples and launch the docker.
2. run nv_run_pretraining.py using the same parameter you run run_pretraining.py. It will produce a model in your checkpoint directory.
3. Assume that the exported model's name is 'bert.onnx'. Run model_transform.py bert.onnx
4. Then run layer_norm_transform.py bert_optimized.onnx. The final model name would be bert_optimized_layer_norm.onnx
5. Now, you can run training with the newly created model.

Note that if you want to change model's configuration, you can edit bert_config.json in the BERT directory.

Example commands:
Step 2 (inside docker):
python3 /workspace/bert/nv_run_pretraining.py --input_dir=data/bookcorpus/hdf5_shards/ --output_dir=/results/checkpoints1 --config_file=bert_config.json --bert_model=bert-large-uncased --warmup_proportion=0 --num_steps_per_checkpoint=2000 --learning_rate=0.875e-4 --seed=42 --do_train --phase2 --max_seq_length=512 --max_predictions_per_seq=80 --max_steps=200 --train_batch_size=2 

Step 3 (inside onnxruntime/build/Linux/RelWithDeb):
sudo /data/anaconda/envs/py35/bin/python /bert_ort/wechi/DeepLearningExamples/PyTorch/LanguageModeling/BERT/model_transform.py /bert_ort/wechi/DeepLearningExamples/PyTorch/LanguageModeling/BERT/results/checkpoints1/bert_for_pretraining_without_loss_vocab_30528_hidden_1024_maxpos_512.onnx 

Step 4 (inside onnxruntime/build/Linux/RelWithDeb):
sudo /data/anaconda/envs/py35/bin/python /bert_ort/wechi/DeepLearningExamples/PyTorch/LanguageModeling/BERT/layer_norm_transform.py /bert_ort/wechi/DeepLearningExamples/PyTorch/LanguageModeling/BERT/results/checkpoints1/bert_for_pretraining_without_loss_vocab_30528_hidden_1024_maxpos_512_optimized.onnx

Step 5 (inside onnxruntime/build/Linux/RelWithDeb):
./onnxruntime_training_bert --num_of_perf_samples=100 --train_batch_size=1 --mode=perf --model_name /bert_ort/wechi/DeepLearningExamples/PyTorch/LanguageModeling/BERT/results/checkpoints1/bert_for_pretraining_without_loss_vocab_30528_hidden_1024_maxpos_512_optimized_layer_norm
