# ORTModule Training Convergence Investigation

## 1. Discovering

Convergence issues can be identified by:
- Large discrepancy on core training metrics including training loss, evaluation loss, model specific AUC metrics.
- Runtime failures (for example loss scaler reach the minimum triggering an exception).

Before looking into further, we should clarify few things (if possible):
- If we change seed for baseline run, whether the metric diff is big?
  (Make sure the discrepancy is not introduced by random)
- What's the very first steps we see obvious diverges?
- Still repro once remove randomness?
	- Set same seeds
	- Set dropout ratio to 0
	- Set compute to be deterministic.


## 2. Collect Activations Summarizer

Add codes:

```diff
+	from onnxruntime.training.ortmodule._runtime_inspector import ActivationSummarizer
+	summarizer = ActivationSummarizer("pt_no_randomness_fulllayer")
+	summarizer.initialize(model)

```
Run training script to the steps that triggered the divergence. A folder named `pt_no_randomness_fulllayer` is created in current working directory. For each step, there is a folder containing summaries for every activation tensor.


Add few lines of code:
```diff
	from onnxruntime.training.ortmodule import ORTModule
	from onnxruntime.training.ortmodule._runtime_inspector import ActivationSummarizer

+	summarizer = ActivationSummarizer("ort_no_randomness_fulllayer")
+	summarizer.initialize(model)
	model = ORTModule(model)
```

> ActivationSummarizer must be initialized before wrapping ORTModule.

Run training script to the steps that triggered the divergence. Similarly, a folder named `ort_no_randomness_fulllayer` is created in current working directory.

Run command to generate per step summary:

Be noted: here we use the topo order of PyTorch to merge activation summary, to make it easier to compare the result.

```bash
python merge_summary.py --path pt_no_randomness_fulllayer --order pt_no_randomness_fulllayer/step_0/order.txt

python merge_summary.py --path ort_no_randomness_fulllayer --order pt_no_randomness_fulllayer/step_0/order.txt
```

Manual diff the generate per-step summary to find the where is the first big diff happens.
