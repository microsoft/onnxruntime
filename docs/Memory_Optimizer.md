# Memory Optimizer for ONNX Runtime Training

## Introduction

ONNX Runtime Training provides a capability trading node/subgraph re-computations for better memory efficiency.
Specifically, a list of re-computable operators is pre-defined, with which memory optimizer graph transformer will iterate the graph to find all re-computable subgraph candidates.

When training with `ORTModule`, by default, the graph transformer will scan the execution graph to find all eligible subgraphs to recompute, along with sizes that can be saved. Users can pick up some of the subgraphs to enable by environment variables.

## When memory optimizer can help?

Classical scenarios include:

- `ORTModule` runs a model with batch size B (for example 2^N), the memory bandwidth and compute are not fully saturated, while it hits OOM to run a bigger batch size (for example 2^(N+1)).

- For big models, `ORTModule` fails to run the minimum allowed batch size, so performance can be compromised for a successful run.

Not all models and recipes need this optimizer technique. Imagine if your training recipe uses a batch size 6 (GPU compute and memory are fully saturated), and you don't need bump it to 8 to maintain a fixed global batch size. Enabling recompute maybe not bring better throughput on batch size 8 than the original batch size 6.

## Usage


Make sure ONNX Runtime training wheel is installed and correctly configured.
Integrate models using `ORTModule`.
```diff
	model = build_model()

+	from onnxruntime.training.ortmodule import ORTModule
+	model = ORTModule(model)
```

There are two modes to enable the memory optimizations:
- Aggressively Recompute All for Transformer Models, enabled by `export ORTMODULE_MEMORY_OPT_LEVEL=1`. This will recompute all detected subgraphs within each Transformer Attention or MLP layer. It is easy to enable, but be noted this recompute plan may NOT be the best one. In this mode, `ORTMODULE_MEMORY_OPT_CONFIG` env values passed by users are not respected.
- User Specified Subgraph Recompute, enabled by `export ORTMODULE_MEMORY_OPT_LEVEL=0` and `export ORTMODULE_MEMORY_OPT_CONFIG=<plan1 config>,<plan2 config>,...`. This is an advanced usage, allows users to find the most suitable graphs to recompute, at the cost of overhead to look for the best plans.

### Mode 1 - Simple Usage (Aggressively Recompute All)


1. Set memory optimization level to be TRANSFORMER_LAYERWISE_RECOMPUTE, by `export ORTMODULE_MEMORY_OPT_LEVEL=1`
2. Run the training as usual; check the logs, you could find something like this:
	```
	Memory Optimizer     :  ON   :  Memory Optimization Level: [TRANSFORMER_LAYERWISE_RECOMPUTE], Optimization Config: [Reshape+Where+:1:-1,BiasSoftmax+:1:-1,Cast+:1:-1,BiasGelu+:1:-1,FusedMatMul+:1:-1,Add+:1:-1,Reshape+Unsqueeze+Unsqueeze+Cast+Sub+Mul+Cast+:1:-1]
									Configs                                              Freq  Max Saving(Bytes)  Saving Symbolic(Bytes)
	- Plan 1            :  ON   :  Reshape+Where+:1:-1                                  1     134,217,728        128.0*inputs_input_ids_dim0*inputs_input_ids_dim1**2
	- Plan 2            :  ON   :  BiasSoftmax+:1:-1                                    1     134,086,656        128.0*inputs_input_ids_dim0*inputs_input_ids_dim1*(inputs_input_ids_dim1 - 1)
	- Plan 3            :  ON   :  Cast+:1:-1                                           1     67,043,328         64.0*inputs_input_ids_dim0*inputs_input_ids_dim1*(inputs_input_ids_dim1 - 1)
	- Plan 4            :  ON   :  BiasGelu+:1:-1                                       1     20,951,040         20480.0*inputs_input_ids_dim0*(inputs_input_ids_dim1 - 1)
	- Plan 5            :  ON   :  FusedMatMul+:1:-1                                    1     20,951,040         20480.0*inputs_input_ids_dim0*(inputs_input_ids_dim1 - 1)
	- Plan 6            :  ON   :  Add+:1:-1                                            1     5,237,760          5120.0*inputs_input_ids_dim0*(inputs_input_ids_dim1 - 1)
	- Plan 7            :  ON   :  Reshape+Unsqueeze+Unsqueeze+Cast+Sub+Mul+Cast+:1:-1  1     4,096              4.0*inputs_input_ids_dim0*inputs_input_ids_dim1
	- Plan 8            :  OFF  :  Cast+:2:-1                                           1     2,048              2.0*inputs_input_ids_dim0*inputs_input_ids_dim1
	```
3. As shown above, `Config` is a string representative for a re-computable subgraph. All are enabled for recompute in this case.


### Mode 2 -  Advanced Usage (User Selected Subgraph Recompute)

1. Be noted `ORTMODULE_MEMORY_OPT_LEVEL` is by default be 0. Run the training as usual; then stop it after training a few steps.
2. Check the logs, you could find something like this:
	```
	Memory Optimizer     :  OFF  :  Enable with env ORTMODULE_MEMORY_OPT_LEVEL=1 or ORTMODULE_MEMORY_OPT_CONFIG=<plan1 config>,<plan2 config>,...
									Configs                                              Freq  Max Saving(Bytes)  Saving Symbolic(Bytes)
	- Plan 1            :  OFF  :  Reshape+Where+:1:-1                                  1     134,217,728        128.0*inputs_input_ids_dim0*inputs_input_ids_dim1**2
	- Plan 2            :  OFF  :  BiasSoftmax+:1:-1                                    1     134,086,656        128.0*inputs_input_ids_dim0*inputs_input_ids_dim1*(inputs_input_ids_dim1 - 1)
	- Plan 3            :  OFF  :  Cast+:1:-1                                           1     67,043,328         64.0*inputs_input_ids_dim0*inputs_input_ids_dim1*(inputs_input_ids_dim1 - 1)
	- Plan 4            :  OFF  :  BiasGelu+:1:-1                                       1     20,951,040         20480.0*inputs_input_ids_dim0*(inputs_input_ids_dim1 - 1)
	- Plan 5            :  OFF  :  FusedMatMul+:1:-1                                    1     20,951,040         20480.0*inputs_input_ids_dim0*(inputs_input_ids_dim1 - 1)
	- Plan 6            :  OFF  :  Add+:1:-1                                            1     5,237,760          5120.0*inputs_input_ids_dim0*(inputs_input_ids_dim1 - 1)
	- Plan 7            :  OFF  :  Reshape+Unsqueeze+Unsqueeze+Cast+Sub+Mul+Cast+:1:-1  1     4,096              4.0*inputs_input_ids_dim0*inputs_input_ids_dim1
	- Plan 8            :  OFF  :  Cast+:2:-1                                           1     2,048              2.0*inputs_input_ids_dim0*inputs_input_ids_dim1
	```
3. As shown above, `Config` is a string representative for a re-computable subgraph. All are disabled for recompute in this case.
4. Set environment variable `ORTMODULE_MEMORY_OPT_CONFIG` to enable some of the subgraphs to do recompute.
	```bash
	# Use comma as a separator for enabling more than one subgraphs.
	export ORTMODULE_MEMORY_OPT_CONFIG="BiasGelu+:1:1"
	# Explanation:
	#  > BiasGelu+ is the subgraph string representative;
	#  > 1 in the middle indicates 'Recompute' is enabled (0, on the contrary indicates it's disabled)
	#  > The last 1 means the initial 1 subgraph occurrences will be recomputed, all others are left as it is, filling `-1` will make all occurrences be recomputed.

	```
5. Then run the training again, and you will see logs like this:
	```
	Memory Optimizer     :  ON   :  Memory Optimization Level: [USER_SPECIFIED], Optimization Config: [BiasGelu+:1:-1]
									Configs                                              Freq  Max Saving(Bytes)  Saving Symbolic(Bytes)
	- Plan 1            :  OFF  :  Reshape+Where+:1:-1                                  1     134,217,728        128.0*inputs_input_ids_dim0*inputs_input_ids_dim1**2
	- Plan 2            :  OFF  :  BiasSoftmax+:1:-1                                    1     134,086,656        128.0*inputs_input_ids_dim0*inputs_input_ids_dim1*(inputs_input_ids_dim1 - 1)
	- Plan 3            :  OFF  :  Cast+:1:-1                                           1     67,043,328         64.0*inputs_input_ids_dim0*inputs_input_ids_dim1*(inputs_input_ids_dim1 - 1)
	- Plan 4            :  ON   :  BiasGelu+:1:-1                                       1     20,951,040         20480.0*inputs_input_ids_dim0*(inputs_input_ids_dim1 - 1)
	- Plan 5            :  OFF  :  FusedMatMul+:1:-1                                    1     20,951,040         20480.0*inputs_input_ids_dim0*(inputs_input_ids_dim1 - 1)
	- Plan 6            :  OFF  :  Add+:1:-1                                            1     5,237,760          5120.0*inputs_input_ids_dim0*(inputs_input_ids_dim1 - 1)
	- Plan 7            :  OFF  :  Reshape+Unsqueeze+Unsqueeze+Cast+Sub+Mul+Cast+:1:-1  1     4,096              4.0*inputs_input_ids_dim0*inputs_input_ids_dim1
	- Plan 8            :  OFF  :  Cast+:2:-1                                           1     2,048              2.0*inputs_input_ids_dim0*inputs_input_ids_dim1
	```
6. You may need iterate a few times on step 4 and 5 until you find a good config for this model to run a bigger batch size. Or you may fail to find if memory optimization does not apply to the model well.

## Optimization Configuration

The basic optimization unit is represented with a unique `cluster id`, for example `BiasGelu+` is one `cluster id`.
Following `cluster id` is the `optimization strategy`: 0 - none, 1 - recompute, 2 - recompute with compromised memory saving.
Following `optimization strategy` is the `request count` to apply the given optimization. Using `-1` to apply all. This would give user a bit more flexibility to avoid unnecessary memory saving.

### Compromised Recompute

If you check the above logs, there is a config `Cast+:2:-1`, `2` indicates it's a recomputation than can save part of the stashed activation size, not all. Recompute the subgraphs under it usually will save part of the activation (for example half of them), not all of them. Follow the same way to enable it.

## Dev Notes

### Memory Optimization Debug Infos

Using following log level
> ort_model = ORTModule(pt_model, DebugOptions(log_level=LogLevel.DEVINFO))

Besides the logs shown in `LogLevel.INFO`, you can also see different node patterns that can apply different optimization options.

The way we get the table:
- For a specific node, it might has different optimization options, we [generates](../orttraining/orttraining/core/optimizer/memory_optimizer/common.h#L124C26-L124C26) a hash (called `Node Cluster ID`) for the node according to all available optimization options.
- Map all nodes having same `Node Cluster ID` in buckets, each bucket is displayed as one row.

```
MemoryInsight Summary - User config: not provided
===========================================================================================================================================
|Freq   | Memory Optimization Opportunities (Clustered by node-level activation patterns)                                                 |
|_ _ _ _|_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _|
|6      |For each row options are mutually exclusive, only one of them can be enabled.                                                    |
|       |                                                                                                                                 |
|       |>>Option 1     : Recompute subgraph FusedMatMul+Add+Reshape+                                                                     |
|       |  Status       : Disabled. Enable with export ORTMODULE_MEMORY_OPT_CONFIG=FusedMatMul+Add+Reshape+:1:-1                          |
|       |  Stashed Activations:                                                                                                           |
|       |   - ReuseFreq :  Output 0(6),                                                                                                   |
|       |   - Output 0  : [((inputs_input_ids_dim0)*(inputs_input_ids_dim1)*(32)*(240))], byte/elem: 2, 100% saved                        |
|_ _ _ _|_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _|
|5      |For each row options are mutually exclusive, only one of them can be enabled.                                                    |
|       |                                                                                                                                 |
|       |>>Option 1     : Recompute subgraph FusedMatMul+                                                                                 |
|       |  Status       : Disabled. Enable with export ORTMODULE_MEMORY_OPT_CONFIG=FusedMatMul+:1:-1                                      |
|       |  Stashed Activations:                                                                                                           |
|       |   - Output 0  : [((inputs_input_ids_dim0)*(inputs_input_ids_dim1)*(10240))], byte/elem: 2, 100% saved                           |
|_ _ _ _|_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _|
|5      |For each row options are mutually exclusive, only one of them can be enabled.                                                    |
|       |                                                                                                                                 |
|       |>>Option 1     : Recompute subgraph Cast+                                                                                        |
|       |  Status       : Disabled. Enable with export ORTMODULE_MEMORY_OPT_CONFIG=Cast+:1:-1                                             |
|       |  Stashed Activations:                                                                                                           |
|       |   - Output 0  : [((inputs_input_ids_dim0)*(32)*(inputs_input_ids_dim1)*(inputs_input_ids_dim1))], byte/elem: 2, 100% saved      |
|_ _ _ _|_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _|
|1      |For each row options are mutually exclusive, only one of them can be enabled.                                                    |
|       |                                                                                                                                 |
|       |>>Option 1     : Recompute subgraph Reshape+Unsqueeze+Unsqueeze+Cast+Sub+Mul+Cast+                                               |
|       |  Status       : Disabled. Enable with export ORTMODULE_MEMORY_OPT_CONFIG=Reshape+Unsqueeze+Unsqueeze+Cast+Sub+Mul+Cast+:1:-1    |
|       |  Stashed Activations:                                                                                                           |
|       |   - Output 0  : [((inputs_input_ids_dim0)*(1)*(1)*(inputs_input_ids_dim1))], byte/elem: 4, 100% saved                           |
|       |                                                                                                                                 |
|       |>>Option 2     : RecomputeWithCompromise subgraph Cast+                                                                          |
|       |  Status       : Disabled. Enable with export ORTMODULE_MEMORY_OPT_CONFIG=Cast+:2:-1                                             |
|       |  Stashed Activations:                                                                                                           |
|       |   - Output 0  : [((inputs_input_ids_dim0)*(1)*(1)*(inputs_input_ids_dim1))], byte/elem: 4, 50% saved                            |
|_ _ _ _|_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _|

```

## Notes

The feature is in the experimental stage, we will tune and refine it according to real use cases.
