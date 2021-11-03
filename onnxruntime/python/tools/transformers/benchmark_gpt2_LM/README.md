- **Benchmark tool**

  This is useful to run end to end testing for any onnx model. For other kind of models, there are some specific change needed.

  **Benchmark Tool :**

  1. Input file format is list of queries per line.

  2. Output file is a tsv which has the following columns in order:

     Total Inference Time : Inference time is the time spent in computation for outputs from inputs with a model.

     Counter : Number of iterations in beam search

     Total Model Time: Total time spent on model (inference + some input/output processing). There is some additional time involved which can be following (needs investigation where though)

     1. The session is run IO binded in onnx, the results seems to copied from GPU to CPU for doing beam search. 
     2. Past state is taken out and copied back for next input. 
     3. Attention mask, positions ids are generated.
     4. 

     Total Search Time: Total time spent in beam search. Once outputs are processed, they are sorted and required number of sequences are only processed further.

     Result: Resulting suggestions for the query

     Total Query Time: E2E time for a query, which includes encoding, decoding, inference and search time.

     

 Is it possible that a model won't have any suggestions, after the beam search - I don't think so

  To run APPG with ONNX:
  $env:ENABLE_ORT=1
  $env:ENABLE_DLIS=''
  $env:ONNX_MODEL_PATH=".\\onnx_model\\deepsuggest_opt.onnx"

  python .\model\calc_appg.py .\NWMeaseurement_RefPredctions_230K.tsv .\output_file .\NWMeaseurement_RefPredctions_230K.tsv

  To run with DLIS :

  $env:ENABLE_DLIS=1
  $env:ONNX_MODEL_PATH=''
  $env:ENABLE_ORT=''
  $env:DS_DIR="E:\\mycode\\DeepsuggestChanged\\Deepsuggest\\"


After any changes run the onnx model with:

>python .\model\main.py -t onnx -m .\onnx_model\deepsuggest_embed_fused_with_pos.onnx -i .\100Prefixes_RandomSet_WithNoRepeat.tsv -o .\100Prefixes_RandomSet_WithNoRepeat_onnx_post_fused_now.tsv
>python .\compare_results.py onnx

  **Benchmark Tool :**

1. Input file format is list of queries per line.

2. Output file is a tsv which has the following columns in order:

Number of iterations of search/sampling (Counter)
Total encoding+decoding time
Total Inference Time
Total Search Time
Total E2E Time
Result

**Tokenizer :**

Currently only GPT2Tokenizer is supported.
The default tokenizer comes from 'model_files/' (probably saved from pytorch), if a custom tokenizer is required pass the path to it.

**Test Data:**

1K set of input queries

10K set of input queries

50K set of input queries



1K and 10K queries are actually the first 1K and 10K queries from the 50K set. The number of inputs is the only difference.



How to test the script:

1. Include an example to test onnx model e2e and what all results does it include

  
  **Accurancy Measurement:**

  calc_appg.py is present to measure the accurancy of the prediction model 

  This tool needs some more refactoring as this is in its crude form.

  python .\model\calc_appg.py -t "onnx" -m .\onnx_model\deepsuggest_embed_fused_with_pos.onnx --num_beams 2 -i ..\DeepSuggest_data\NWMeasurement_RefPredictions_230K.tsv -o calc_output_file.tsv -r ..\DeepSuggest_data\NWMeasurement_RefPredictions_230K.tsv


Made output file optional. 
