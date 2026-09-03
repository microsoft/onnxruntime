**Benchmark tool**

This can be used to test beam search on any gpt2 model. Both pytorch and onnx models can be tested.

Other kind of models are not supported yet.

Also, this script defaults to GPU. To test a model on CPU pass --device 'cpu' when running the model.



**Requirements:**

1. GPU machine. V100 was used to test this. 

2. GPT2 ONNX Model that follows the regular GPT2 style of inputs and outputs.

   Model should have 4 inputs, with the following dimensions:

   **input_ids** `[batch_size,seq_len]`

   **position_ids** `[batch_size,seq_len]`

   **attention_mask** `[batch_size,total_seq_len]`

   **past_0**`[2,batch_size,16,past_seq_len,64]`

   **past_1** and so on with the same structure

   And 2 outputs:

   **logits**	type: `[batch_size,seq_len,50297]`

   **present_0** type: `[2,batch_size,16,total_seq_len,64]`

   **present_1** and so on with the same structure

   past_0, attention_mask, logits, present_0 can be float16 or float32. inputs_ids and position_ids are int64.

   

3. Create python venv with all the packages in required_packages.txt

   >  pip install -r requirements.txt

   

**Run the script:**

```
Test onnx:

the model onnx_gpt2.onnx has to be converted using onnxruntime.transformers.convert_to_onnx if not already done.

python .\model\main.py -t "onnx" -m "onnx_gpt2.onnx" --num_beams 2 --run_beam_search --length_penalty 1.6 --tokenizer gpt2  --input_file .\1K_queries.tsv --output_file .\1K_queries_onnx_output.tsv

 
Test pytroch model:

python .\model\main.py -t pt -m "gpt2" --num_beams 2 --run_beam_search --length_penalty 1.6 --tokenizer gpt2  --input_file .\1K_queries.tsv  --output_file .\1K_queries_pt_output.tsv

Note: 
1. Provide absolute paths for all the files.
2. --device 'cpu' for CPU.

```

The following options are required to run the script as provided in the above example:

1. -t "onnx"
2. -m <path to the model location> if not one of the standard HF models.
3. --run_beam_search - mandatory currently but left as an option as without this needs support soon.
4. --num_beams <number of beams to explore> Higher the number, higher the convergence time would be 
5. --tokenizer <path to tokenizer> - currently a tokenizer is provided, if a custom tokenizer needs to be used, might need some changes.
6. --input_file <input file with query per line>: 	Sample is provided in the repo, refer 1K_ queries.tsv
7. --output_file <output file with the results> : Following is the format of output.tsv

| TotalInferenceTime | Counter | TotalModelTime | TotalSearchTime | Result                                                       | TotalQueryTime |
| ------------------ | ------- | -------------- | --------------- | ------------------------------------------------------------ | -------------- |
| 5.2609             | 4       | 6.9058         | 7.4306          | ["snap", "snake", "snacks",  "snack", "snap on", "snakes",  "snapped", "snail"] | 19.6145        |
| 11.583             | 8       | 15.8132        | 17.6436         | ["runoff", "runoff test", "runoff test  kit", "runoff test kit/", "runoff test kit/youtube",  "runoff test kit/youtube/", "runoff test kit/youtube/h",  "runoff test kit/youtube/hc"] | 36.4981        |

**Total Inference Time** : Inference time is the time spent in computation for outputs from inputs with a model.

**Counter** : Number of iterations in beam search

**Total Model Time**: Total time spent on model (inference + some input/output processing). There is some additional time involved which can be following 

**Total Search Time**: Total time spent in beam search. Once outputs are processed, they are sorted and required number of sequences are only processed further.

**Result:** Resulting suggestions for the query

**Total Query Time:** E2E time for a query, which includes encoding, decoding, inference and search time.



**Test Data included in the repo:**

1K set of input queries

10K set of input queries
