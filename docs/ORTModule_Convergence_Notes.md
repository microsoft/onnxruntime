# ORTModule Training Convergence Investigation

## 1. Discovering

Convergence issues can be identified by:
- Large discrepancies in core training metrics including training loss, evaluation loss, model specific AUC metrics.
- Runtime failures (for example when the loss scaler reaches the minimum, triggering an exception).

Before looking into this further, we should clarify a few things (if possible):
- If we change the seed for the baseline run, whether the metric diff is big?
  (Make sure the discrepancy is not introduced by randomness)
- What are the very first steps we see obvious divergence?
- Still reproducible once randomness is removed?
- Set same seeds
- Set the dropout ratio to 0
- Set compute to be deterministic and torch-comparable (TODO(pengwa): need a flag for this).


## 2. Collect Activation Statistics

### Add a few lines of code, run script to collect statistics:

<table>
<tr>
<th>Baseline</th>
<th>ORTModule</th>
</tr>
<tr>
<td>
<sub>

```diff
+ from onnxruntime.training.utils.hooks import SubscriberManager,
+                                              StatisticsSubscriber
+ sub_m = SubscriberManager()
+ sub_m.subscribe(model, [StatisticsSubscriber(output_dir="pt_out",
+                                              override_output_dir=True)])
```

</sub>
</td>
<td>
<sub>

```diff
model = ORTModule(model)
+ from onnxruntime.training.utils.hooks import SubscriberManager,
+                                              StatisticsSubscriber
+ sub_m = SubscriberManager()
+ sub_m.subscribe(model, [StatisticsSubscriber(output_dir="ort_out",
+                                              override_output_dir=True)])
```

</sub>
</td>
</tr>

<tr>
<td>

- Run training script to the steps that trigger the divergence.
- A folder named `pt_out` is created in the current working directory.
- For each step, there is a folder containing summaries for every activation tensor.

</td>
<td>


- Run training script to the steps that trigger the divergence.
- Similarly, a folder named `ort_out` is created in the current working directory.
- `StatisticsSubscriber` can be subscribed before OR after wrapping ORTModule.

</td>
</tr>
</table>


Arguments:
- output_dir: the directory in all activation statistics files will be stored.
- start_step [optional]: the first step that runs subscriber actions.
- end_step [optional]: the end step (exclusively) that runs subscriber actions.
- override_output_dir: whether `output_dir` can be overridden if it already exists.

Check [StatisticsSubscriber implementation](../orttraining/orttraining/python/training/utils/hooks/_statistics_subscriber.py)  for more information.

### Run command to generate per-step summary

```bash
python -m onnxruntime.training.utils.hooks.merge_activation_summary --pt_dir pt_out --ort_dir ort_out --output_dir /tmp/output
```

### Manually compare the generated per-step summary to find the first big diff.
