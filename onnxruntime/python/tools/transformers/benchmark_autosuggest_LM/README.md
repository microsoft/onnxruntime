- **Benchmark tool**

  This is useful to run end to end testing for any onnx model. For other kind of models, there are some specific changed needed.

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

     

  To run APPG with ONNX:
  $env:ENABLE_ORT=1
  $env:ENABLE_DLIS=''
  $env:ONNX_MODEL_PATH=".\\onnx_model\\deepsuggest_opt.onnx"

  python .\model\calc_appg.py .\NWMeaseurement_RefPredctions_230K.tsv .\output_file .\NWMeaseurement_RefPredctions_230K.tsv

  To run with DLIS :

  $env:ENABLE_DLIS=1
  $env:ONNX_MODEL_PATH=''
  $env:ENABLE_ORT=''
  $env:DS_DIR="E:\\mycode\\Deepsuggest\\Deepsuggest\\"

  

  