**Benchmark tool**

This can be used to test beam search on 1 layer gpt2 onnx model.

Other kind of models are not supported yet.



**Requirements:**

1. GPU machine. V100 was used to test this. 

2. GPT2 ONNX Model that follows the regular GPT2 style of inputs and outputs.

    Model should have 4 inputs, with the following dimensions: 

   **input_ids** `[batch_size,seq_len]`

   **position_ids** `[batch_size,seq_len]`

   **attention_mask** `[batch_size,total_seq_len]`

   **past_0**`[2,batch_size,16,past_seq_len,64]`

   And 2 outputs:

   **logits**	type: `[batch_size,seq_len,50297]`

   **present_0** type: `[2,batch_size,16,total_seq_len,64]`

   past_0, attention_mask, logits, present_0 can be float16 or float32. inputs_ids and position_ids are int64.

   

3. Create python venv with all the packages in required_packages.txt

   >  pip install -r required_packages.txt

   

**Run the script:**

```
python .\model\main.py -t "onnx" -m .\onnx_model\deepsuggest_embed_fused_with_pos.onnx --num_beams 2 -i .\1K_queries.tsv -o .\1K_queries_onnx_post_fused_now.tsv
```

The following options are required to run the script as provided in the above example:

1. -t "onnx"
2. -m <path to the model location>
3. --num_beams <beam_size for iterations>
4. --input_file <input file with query per line>: 	Sample is provided in the repo, refer 1K_ queries.tsv
5. --output_file <output file with the results> : Following the format of output.tsv

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



**Tokenizer :**

Currently only GPT2Tokenizer is supported. The one available in the repo is under tokenizer_files\ - it is exactly same apart from the extra tokens that it uses for this particular model.

The default tokenizer comes from 'model_files/' , if a custom tokenizer is required pass the path to it. There might be some changes needed for other tokenizers to work.





**Test Data included in the repo:**

1K set of input queries

10K set of input queries




 **Accuracy Measurement:**

TBD.
