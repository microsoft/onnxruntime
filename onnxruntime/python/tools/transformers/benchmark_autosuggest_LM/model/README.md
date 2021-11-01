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

