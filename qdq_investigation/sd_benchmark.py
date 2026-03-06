"""
Stable Diffusion Component Latency Benchmark

Benchmarks latency of SD pipeline component models for three sets:
  - float (dynamic shapes)
  - qdq (quantized, qdq cleanup disabled)
  - qdq_cleanup (quantized, qdq cleanup enabled via
    session.enable_quant_qdq_cleanup = "1")

Only directories that contain a `model.onnx` are benchmarked.

For each setting the ORT-optimized graph is also dumped (separately from
latency measurement) and analysed with onnx-ir to produce per-component
operator counts, including a breakdown of QuantizeLinear / DequantizeLinear
nodes by activation vs. weight (initializer) first-input.

Output is a nested dictionary structure:
{
  "latency_ms": {
	"float": { "unet": { ...stats... }, ... },
	"qdq":   { ... },
	"qdq_cleanup": { ... }
  },
  "op_analysis": {
	"float": {
	  "unet": {
		"total_ops": 123,
		"op_counts": { "Conv": 10, ... },
		"total_q": 0, "total_dq": 0,
		"q_activation": 0, "q_weight": 0,
		"dq_activation": 0, "dq_weight": 0
	  }, ...
	},
	"qdq":   { ... },
	"qdq_cleanup": { ... }
  },
  "metadata": { ... }
}
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import shutil
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
	import onnxruntime as ort
except ImportError:
	print("Error: onnxruntime not installed. Run: pip install onnxruntime")
	sys.exit(1)

try:
	import onnx_ir as ir
except ImportError:
	print("Error: onnx-ir not installed. Run: pip install onnx-ir")
	sys.exit(1)


DEFAULT_WARMUP = 1
DEFAULT_ITERATIONS = 5


DYNAMIC_DIM_MAP: Dict[str, int] = {
	"batch": 1,
	"sequence": 77,
	"unet_sample_batch": 1,
	"unet_sample_channels": 4,
	"unet_sample_height": 64,
	"unet_sample_width": 64,
	"unet_time_batch": 1,
	"unet_hidden_batch": 1,
	"unet_hidden_sequence": 77,
	"decoder_batch": 1,
	"decoder_channels": 4,
	"decoder_height": 64,
	"decoder_width": 64,
	"encoder_batch": 1,
	"encoder_channels": 3,
	"encoder_height": 512,
	"encoder_width": 512,
	"Addlatent_sample_dim_0": 1,
	"Addlatent_sample_dim_1": 4,
	"Addlatent_sample_dim_2": 64,
	"Addlatent_sample_dim_3": 64,
}


@dataclass
class BenchmarkConfig:
	model_root: str
	warmup: int = DEFAULT_WARMUP
	iterations: int = DEFAULT_ITERATIONS
	output_file: Optional[str] = None
	seed: int = 42
	verbose: bool = False
	skip_latency: bool = False
	skip_analysis: bool = False
	optimized_model_dir: Optional[str] = None


def inspect_model_inputs(session: ort.InferenceSession) -> List[Dict[str, Any]]:
	inputs = []
	for inp in session.get_inputs():
		inputs.append(
			{
				"name": inp.name,
				"shape": inp.shape,
				"type": inp.type,
			}
		)
	return inputs


def numpy_dtype_from_onnx_type(onnx_type: str) -> np.dtype:
	type_map = {
		"tensor(float)": np.float32,
		"tensor(float16)": np.float16,
		"tensor(double)": np.float64,
		"tensor(int64)": np.int64,
		"tensor(int32)": np.int32,
		"tensor(int16)": np.int16,
		"tensor(int8)": np.int8,
		"tensor(uint8)": np.uint8,
		"tensor(bool)": np.bool_,
	}
	return type_map.get(onnx_type, np.float32)


def resolve_dynamic_dim(
	dim: Any,
	input_name: str,
	dynamic_dim_map: Dict[str, int],
) -> int:
	if isinstance(dim, int):
		return dim

	if dim is None:
		dim_str = ""
	else:
		dim_str = str(dim)

	if dim_str in dynamic_dim_map:
		return dynamic_dim_map[dim_str]

	dim_lower = dim_str.lower()

	if dim_lower in dynamic_dim_map:
		return dynamic_dim_map[dim_lower]

	raise ValueError(
		f"Unknown dynamic dim '{dim_str or '<None>'}' for input '{input_name}'. "
		"Add it to DYNAMIC_DIM_MAP."
	)


def resolve_shape(
	shape: Sequence[Any],
	input_name: str,
	dynamic_dim_map: Dict[str, int],
) -> List[int]:
	resolved: List[int] = []
	for dim in shape:
		resolved.append(resolve_dynamic_dim(dim, input_name, dynamic_dim_map))
	return resolved


def create_input_tensor(input_name: str, dtype: np.dtype, shape: List[int]) -> np.ndarray:
	name_lower = input_name.lower()

	if np.issubdtype(dtype, np.integer):
		if "input_ids" in name_lower:
			return np.random.randint(0, 49408, size=shape, dtype=dtype)
		return np.zeros(shape, dtype=dtype)

	if np.issubdtype(dtype, np.bool_):
		return np.ones(shape, dtype=dtype)

	return np.random.randn(*shape).astype(dtype)


def generate_inputs(
	session: ort.InferenceSession,
	dynamic_dim_map: Dict[str, int],
) -> Dict[str, np.ndarray]:
	inputs: Dict[str, np.ndarray] = {}

	for info in inspect_model_inputs(session):
		input_name = info["name"]
		resolved_shape = resolve_shape(
			info["shape"], input_name, dynamic_dim_map
		)
		dtype = numpy_dtype_from_onnx_type(info["type"])
		inputs[input_name] = create_input_tensor(input_name, dtype, resolved_shape)

	return inputs


def create_session(
	model_path: str,
	extra_session_options: Optional[Dict[str, str]] = None,
) -> ort.InferenceSession:
	sess_options = ort.SessionOptions()
	sess_options.log_severity_level = 3
	if extra_session_options:
		for key, value in extra_session_options.items():
			sess_options.add_session_config_entry(key, value)
	return ort.InferenceSession(
		model_path,
		sess_options,
		providers=["CPUExecutionProvider"],
	)


def run_latency(
	session: ort.InferenceSession,
	inputs: Dict[str, np.ndarray],
	warmup: int,
	iterations: int,
) -> List[float]:
	output_names = [o.name for o in session.get_outputs()]

	for _ in range(warmup):
		_ = session.run(output_names, inputs)

	latencies_ms: List[float] = []
	for _ in range(iterations):
		start = time.perf_counter()
		_ = session.run(output_names, inputs)
		end = time.perf_counter()
		latencies_ms.append((end - start) * 1000.0)

	return latencies_ms


def latency_stats(latencies_ms: List[float]) -> Dict[str, float]:
	arr = np.array(latencies_ms, dtype=np.float64)
	return {
		"mean_ms": round(float(np.mean(arr)), 3),
		"std_ms": round(float(np.std(arr)), 3),
		"min_ms": round(float(np.min(arr)), 3),
		"max_ms": round(float(np.max(arr)), 3),
		"p50_ms": round(float(np.percentile(arr, 50)), 3),
		"p95_ms": round(float(np.percentile(arr, 95)), 3),
		"p99_ms": round(float(np.percentile(arr, 99)), 3),
	}


def find_component_models(set_dir: str) -> List[Tuple[str, str]]:
	"""
	Recursively find component folders that contain model.onnx.

	Returns list of tuples: (component_key, model_path)
	where component_key is relative folder path from set_dir.
	"""
	found: List[Tuple[str, str]] = []
	for root, _, files in os.walk(set_dir):
		if "model.onnx" in files:
			component_key = os.path.relpath(root, set_dir).replace("\\", "/")
			found.append((component_key, os.path.join(root, "model.onnx")))

	found.sort(key=lambda item: item[0])
	return found


def benchmark_component(
	set_name: str,
	component_key: str,
	model_path: str,
	warmup: int,
	iterations: int,
	dynamic_dim_map: Dict[str, int],
	verbose: bool,
	extra_session_options: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
	if verbose:
		print(f"  [{set_name}] {component_key}: loading {model_path}")

	session = create_session(model_path, extra_session_options)
	inputs = generate_inputs(session, dynamic_dim_map)
	latencies = run_latency(session, inputs, warmup, iterations)
	stats = latency_stats(latencies)

	result: Dict[str, Any] = {
		"iterations": iterations,
		"warmup": warmup,
		"model_path": model_path,
		"inputs": {
			k: {
				"shape": list(v.shape),
				"dtype": str(v.dtype),
			}
			for k, v in inputs.items()
		},
	}
	result.update(stats)
	return result


# Maps set_name -> (model_subfolder, extra session options)
_SET_CONFIG: Dict[str, Tuple[str, Optional[Dict[str, str]]]] = {
	"float": ("float", None),
	"qdq": ("qdq", {"session.enable_quant_qdq_cleanup": "0"}),
	"qdq_cleanup": ("qdq", {"session.enable_quant_qdq_cleanup": "1"}),
}


def benchmark_all_components(config: BenchmarkConfig) -> Dict[str, Any]:
	root = os.path.abspath(config.model_root)

	output: Dict[str, Any] = {
		"latency_ms": {name: {} for name in _SET_CONFIG},
		"metadata": {
			"model_root": root,
			"warmup": config.warmup,
			"iterations": config.iterations,
			"seed": config.seed,
			"ort_version": ort.__version__,
			"platform": platform.platform(),
			"python_version": platform.python_version(),
		},
	}

	for set_name, (subfolder, extra_opts) in _SET_CONFIG.items():
		set_dir = os.path.join(root, subfolder)
		if not os.path.isdir(set_dir):
			print(f"Warning: set folder not found, skipping: {set_dir}")
			continue

		components = find_component_models(set_dir)
		if not components:
			print(f"Warning: no model.onnx found under: {set_dir}")
			continue

		print(f"\nBenchmarking set '{set_name}' ({len(components)} components)")
		for component_key, model_path in components:
			print(f"- {component_key}")
			component_result = benchmark_component(
				set_name=set_name,
				component_key=component_key,
				model_path=model_path,
				warmup=config.warmup,
				iterations=config.iterations,
				dynamic_dim_map=DYNAMIC_DIM_MAP,
				verbose=config.verbose,
				extra_session_options=extra_opts,
			)
			output["latency_ms"][set_name][component_key] = component_result

	return output


# ---------------------------------------------------------------------------
# Optimized-graph operator analysis (separate from latency measurement)
# ---------------------------------------------------------------------------


def dump_optimized_model(
	model_path: str,
	extra_session_options: Optional[Dict[str, str]] = None,
	output_dir: Optional[str] = None,
	model_name: str = "model_optimized",
) -> Tuple[str, str, bool]:
	"""Create an ORT session that writes the optimized graph to a directory.

	If *output_dir* is given the optimized model is written there and kept;
	otherwise a temporary directory is created.

	Returns (dir_path, optimized_model_path, is_temp).  When *is_temp* is
	True the caller is responsible for removing *dir_path*.
	"""
	if output_dir is not None:
		os.makedirs(output_dir, exist_ok=True)
		out_dir = output_dir
		is_temp = False
	else:
		out_dir = tempfile.mkdtemp(prefix="sd_bench_opt_")
		is_temp = True

	opt_path = os.path.join(out_dir, f"{model_name}.onnx")

	so = ort.SessionOptions()
	so.log_severity_level = 3
	so.optimized_model_filepath = opt_path
	so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
	so.add_session_config_entry(
		"session.optimized_model_external_initializers_file_name",
		f"{model_name}.onnx.data",
	)
	if extra_session_options:
		for key, value in extra_session_options.items():
			so.add_session_config_entry(key, value)

	# Creating the session triggers the optimized-model dump.
	ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
	return out_dir, opt_path, is_temp


_SHAPE_OPS = frozenset({
	"Reshape", "Transpose", "Squeeze", "Unsqueeze", "Flatten", "Expand",
})


def analyze_optimized_graph(optimized_path: str) -> Dict[str, Any]:
	"""Load an optimized ONNX model with onnx-ir and compute op statistics."""
	model = ir.load(optimized_path)
	graph = model.graph

	op_counts: Dict[str, int] = {}
	total_q = 0
	total_dq = 0
	q_activation = 0
	q_weight = 0
	dq_activation = 0
	dq_weight = 0
	# Q outputs consumed by a shape-related op (Reshape, Transpose, …)
	q_consumed_by_shape_op = 0
	# DQ whose first input is produced by a shape-related op
	dq_input_from_shape_op = 0

	for node in graph:
		op_type = node.op_type
		op_counts[op_type] = op_counts.get(op_type, 0) + 1

		if op_type in ("QuantizeLinear", "DequantizeLinear"):
			# First input determines activation vs weight/initializer.
			first_input = node.inputs[0] if node.inputs else None
			is_weight = first_input is not None and first_input.is_initializer()

			if op_type == "QuantizeLinear":
				total_q += 1
				if is_weight:
					q_weight += 1
				else:
					q_activation += 1
				# Check if any consumer of this Q is a shape-related op.
				for output in node.outputs:
					if output is not None and any(
						c.op_type in _SHAPE_OPS for c in output.consumers()
					):
						q_consumed_by_shape_op += 1
						break
			else:
				total_dq += 1
				if is_weight:
					dq_weight += 1
				else:
					dq_activation += 1
				# Check if the producer of the first input is a shape-related op.
				if first_input is not None:
					producer = first_input.producer()
					if producer is not None and producer.op_type in _SHAPE_OPS:
						dq_input_from_shape_op += 1

	return {
		"total_ops": len(list(graph)),
		"op_counts": dict(sorted(op_counts.items(), key=lambda x: -x[1])),
		"total_q": total_q,
		"total_dq": total_dq,
		"q_activation": q_activation,
		"q_weight": q_weight,
		"dq_activation": dq_activation,
		"dq_weight": dq_weight,
		"q_consumed_by_shape_op": q_consumed_by_shape_op,
		"dq_input_from_shape_op": dq_input_from_shape_op,
	}


def analyze_component(
	set_name: str,
	component_key: str,
	model_path: str,
	verbose: bool,
	extra_session_options: Optional[Dict[str, str]] = None,
	optimized_model_dir: Optional[str] = None,
) -> Dict[str, Any]:
	if verbose:
		print(f"  [{set_name}] {component_key}: dumping optimized graph")

	# Build an informative model name: e.g. "qdq_cleanup__unet" or "float__vae_decoder"
	safe_component = component_key.replace("/", "_").replace("\\", "_")
	model_name = f"{set_name}__{safe_component}"

	out_dir, opt_path, is_temp = dump_optimized_model(
		model_path, extra_session_options,
		output_dir=optimized_model_dir,
		model_name=model_name,
	)
	try:
		result = analyze_optimized_graph(opt_path)
	finally:
		if is_temp:
			shutil.rmtree(out_dir, ignore_errors=True)

	result["model_path"] = model_path
	return result


def analyze_all_components(config: BenchmarkConfig) -> Dict[str, Dict[str, Any]]:
	"""Dump optimized graphs and collect op statistics for all sets."""
	root = os.path.abspath(config.model_root)
	analysis: Dict[str, Dict[str, Any]] = {name: {} for name in _SET_CONFIG}

	for set_name, (subfolder, extra_opts) in _SET_CONFIG.items():
		set_dir = os.path.join(root, subfolder)
		if not os.path.isdir(set_dir):
			continue

		components = find_component_models(set_dir)
		if not components:
			continue

		print(f"\nAnalysing optimized graphs for '{set_name}' ({len(components)} components)")
		for component_key, model_path in components:
			print(f"  - {component_key}")
			analysis[set_name][component_key] = analyze_component(
				set_name=set_name,
				component_key=component_key,
				model_path=model_path,
				verbose=config.verbose,
				extra_session_options=extra_opts,
				optimized_model_dir=config.optimized_model_dir,
			)

	return analysis


def save_results(data: Dict[str, Any], output_file: str) -> str:
	output_file = os.path.abspath(output_file)
	output_dir = os.path.dirname(output_file)
	if output_dir:
		os.makedirs(output_dir, exist_ok=True)

	with open(output_file, "w", encoding="utf-8") as f:
		json.dump(data, f, indent=2)

	return output_file


def default_output_path(model_root: str) -> str:
	script_dir = os.path.dirname(os.path.abspath(__file__))
	model_name = os.path.basename(os.path.abspath(model_root.rstrip("\\/")))
	if not model_name:
		model_name = "sd"
	return os.path.join(script_dir, "results", f"{model_name}_components_latency.json")


def parse_args() -> BenchmarkConfig:
	parser = argparse.ArgumentParser(
		description="Benchmark SD component latency for float, qdq, and qdq_cleanup model sets"
	)
	parser.add_argument(
		"--model-root",
		"-m",
		required=True,
		help="Path to SD root folder that contains 'float' and 'qdq' subfolders",
	)
	parser.add_argument(
		"--warmup",
		"-w",
		type=int,
		default=DEFAULT_WARMUP,
		help=f"Warmup iterations (default: {DEFAULT_WARMUP})",
	)
	parser.add_argument(
		"--iterations",
		"-i",
		type=int,
		default=DEFAULT_ITERATIONS,
		help=f"Benchmark iterations (default: {DEFAULT_ITERATIONS})",
	)
	parser.add_argument(
		"--output",
		"-o",
		default=None,
		help="Output JSON path (default: results/<model_root_name>_components_latency.json)",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Random seed for reproducibility (default: 42)",
	)
	parser.add_argument(
		"--verbose",
		"-v",
		action="store_true",
		help="Verbose logging",
	)
	parser.add_argument(
		"--skip-latency",
		action="store_true",
		help="Skip the latency benchmark pass",
	)
	parser.add_argument(
		"--skip-analysis",
		action="store_true",
		help="Skip the optimized-graph operator analysis pass",
	)
	parser.add_argument(
		"--optimized-model-dir",
		default=None,
		help="Directory to save optimized models in (kept after run). "
		"If not set, a temp directory is used and cleaned up.",
	)

	args = parser.parse_args()
	return BenchmarkConfig(
		model_root=args.model_root,
		warmup=args.warmup,
		iterations=args.iterations,
		output_file=args.output,
		seed=args.seed,
		verbose=args.verbose,
		skip_latency=args.skip_latency,
		skip_analysis=args.skip_analysis,
		optimized_model_dir=args.optimized_model_dir,
	)


def main() -> None:
	config = parse_args()

	if not os.path.isdir(config.model_root):
		print(f"Error: model root folder not found: {config.model_root}")
		sys.exit(1)

	if config.skip_latency and config.skip_analysis:
		print("Error: --skip-latency and --skip-analysis cannot both be set")
		sys.exit(1)

	np.random.seed(config.seed)

	results: Dict[str, Any] = {
		"metadata": {
			"model_root": os.path.abspath(config.model_root),
			"seed": config.seed,
			"ort_version": ort.__version__,
			"platform": platform.platform(),
			"python_version": platform.python_version(),
		},
	}

	# --- Latency benchmark ---
	if not config.skip_latency:
		start = time.perf_counter()
		bench = benchmark_all_components(config)
		elapsed_s = time.perf_counter() - start
		results["latency_ms"] = bench["latency_ms"]
		results["metadata"].update(bench["metadata"])
		results["metadata"]["total_elapsed_s"] = round(elapsed_s, 3)

	# --- Operator analysis (separate pass, does not affect latency) ---
	if not config.skip_analysis:
		results["op_analysis"] = analyze_all_components(config)

	output_path = config.output_file or default_output_path(config.model_root)
	saved_path = save_results(results, output_path)

	print("\nDone!")
	print(f"Saved: {saved_path}")


if __name__ == "__main__":
	main()
