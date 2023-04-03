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
	- Set compute to be deterministic and torch-comparable (TODO(pengwa): need a flag for this).


## 2. Collect Activation Statistics

Add codes:

```diff
+	from onnxruntime.training.utils.hooks import SubscriberManager, StatisticsSubscriber
+	SubscriberManager.subscribe(model, [StatisticsSubscriber("pt_out", override_output_dir=True)])

```
Run training script to the steps that triggered the divergence. A folder named `pt_out` is created in current working directory. For each step, there is a folder containing summaries for every activation tensor.


Add few lines of code:
```diff
	from onnxruntime.training.ortmodule import ORTModule
	from onnxruntime.training.utils.hooks import SubscriberManager, StatisticsSubscriber
	model = ORTModule(model)
+	SubscriberManager.subscribe(model, [StatisticsSubscriber("ort_out", override_output_dir=True)])
```

> `StatisticsSubscriber` can be initialized before OR after wrapping ORTModule.

Run training script to the steps that triggered the divergence. Similarly, a folder named `ort_out` is created in current working directory.

Run command to generate per step summary

```bash
python -m onnxruntime.training.utils.hooks.merge_activation_summary --pt_dir pt_out --ort_dir ort_out --output_dir /tmp/output
```

Manual diff the generate per-step summary to find the where is the first big diff happens.
