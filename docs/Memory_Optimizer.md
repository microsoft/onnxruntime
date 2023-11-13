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

## Quick trial

1. Make sure ONNX Runtime training wheel is installed and correctly configured.
2. Integrate models using `ORTModule`, be noted log_level should be equal to or lower than DEVINFO.
	> ort_model = ORTModule(pt_model, DebugOptions(log_level=LogLevel.DEVINFO))
3. Run the training as usual and redirect all outputs into the log file; then stop it after training a few steps.
4. Check the logging file, and search "Summary", you could find something like this:
	```
	MemoryOptimizer Summary:
	User config:

	=================================
	########Recompute########
	Subgraph: CumSum+Sub+Mul+Unsqueeze+Cast+Mul+Cast+Reshape+Mul+FusedMatMul+Add+Reshape+Cast+Where+Softmax+
		OptimizationType: Disabled
		Patterns:
		PatternShape:input_ids_dim0 x 16 x input_ids_dim1 x input_ids_dim1 x  Frequency:23
	--------------------------------
	Subgraph: FastGelu+
		OptimizationType: Disabled
		Patterns:
		PatternShape:input_ids_dim0 x input_ids_dim1 x 4096 x   Frequency:24
	=================================
	########RecomputeWithCompromise########
	Subgraph: Cast+Where+Softmax+
		OptimizationType: Disabled
		Patterns:
		PatternShape:input_ids_dim0 x 16 x input_ids_dim1 x input_ids_dim1 x  Frequency:24
	--------------------------------
	=================================
	```
5. As shown above, 'Subgraph' shows 1) a string representative for a re-computable subgraph; and 2) current status of memory optimization. All are disabled for recompute in this case.
6. Set environment variable `ORTMODULE_MEMORY_OPT_CONFIG` to enable some of the subgraph to do recompute. In below example, 12 FastGelu related subgraphs are allowed to recompute.
`FastGelu+` is the subgraph string representative; `1` in the middle indicates 'Recompute' is enabled (0, on the contrary indicates it's disabled); `12` means the initial 12 subgraph occurrences will be recomputed, all others are left as it is, filling `-1` will make all occurrences be recomputed.
	```
	export ORTMODULE_MEMORY_OPT_CONFIG="FastGelu+:1:12"
	```
7. Then run the training again, you will see logs like this:
	```
	MemoryOptimizer Summary:
	User config:
	**FastGelu+:1:12**
	=================================
	########Recompute########
	Subgraph: CumSum+Sub+Mul+Unsqueeze+Cast+Mul+Cast+Reshape+Mul+FusedMatMul+Add+Reshape+Cast+Where+Softmax+
		OptimizationType: Disabled
		Patterns:
		PatternShape:input_ids_dim0 x 16 x input_ids_dim1 x input_ids_dim1 x  Frequency:23
	--------------------------------
	Subgraph: FastGelu+
		OptimizationType: **Recompute (requested_count=12, actual applied_count=12)**
		Patterns:
		PatternShape:input_ids_dim0 x input_ids_dim1 x 4096 x   Frequency:24
	=================================
	########RecomputeWithCompromise########
	Subgraph: Cast+Where+Softmax+
		OptimizationType: Disabled
		Patterns:
		PatternShape:input_ids_dim0 x 16 x input_ids_dim1 x input_ids_dim1 x  Frequency:24
	--------------------------------
	=================================
	```
8. You may need iterate few times on step 6 and 7 until you find a good config for this model to run a bigger batch size. Or you may fail to find if memory optimization does not apply to the model well.

## Compromised Recompute

If you check the above logs, there is a separate section called "RecomputeWithCompromise". Recompute the subgraphs under it usually will save part of the activation (for example half of them), not all of them. Follow the same way to enable it.

## Notes

The feature is in experimental stage, we will tune and refine it according to real use cases.
