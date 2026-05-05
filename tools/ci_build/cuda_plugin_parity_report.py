#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
CUDA EP Plugin Registration Parity Report

Compares kernel registrations between the bundled CUDA EP and the plugin CUDA EP
by statically parsing source files or interrogating the kernel registry at runtime.
Produces a report showing which ops are in both builds, only in bundled, or only in plugin.

Usage:
    # Static parse mode (default):
    python tools/ci_build/cuda_plugin_parity_report.py [--repo-root /path/to/onnxruntime]

    # Runtime inquiry mode:
    python tools/ci_build/cuda_plugin_parity_report.py --runtime [--plugin-ep-lib /path/to/libonnxruntime_providers_cuda_plugin.so]
"""

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# Regex patterns for kernel registration macros
# These macros define kernel classes and are the source of truth for op registrations.
KERNEL_EX_PATTERNS = [
    # ONNX_OPERATOR_KERNEL_EX(name, domain, ver, provider, builder, ...)
    re.compile(
        r"ONNX_OPERATOR_KERNEL_EX\s*\(\s*"
        r"(\w+)\s*,\s*"  # name
        r"(\w+)\s*,\s*"  # domain
        r"(\d+)\s*,\s*"  # version
        r"(\w+)\s*,"  # provider
    ),
    # ONNX_OPERATOR_TYPED_KERNEL_EX(name, domain, ver, type, provider, builder, ...)
    re.compile(
        r"ONNX_OPERATOR_TYPED_KERNEL_EX\s*\(\s*"
        r"(\w+)\s*,\s*"  # name
        r"(\w+)\s*,\s*"  # domain
        r"(\d+)\s*,\s*"  # version
        r"(\w+)\s*,\s*"  # type
        r"(\w+)\s*,"  # provider
    ),
    # ONNX_OPERATOR_VERSIONED_KERNEL_EX(name, domain, start_ver, end_ver, provider, builder, ...)
    re.compile(
        r"ONNX_OPERATOR_VERSIONED_KERNEL_EX\s*\(\s*"
        r"(\w+)\s*,\s*"  # name
        r"(\w+)\s*,\s*"  # domain
        r"(\d+)\s*,\s*"  # start_version
        r"(\d+)\s*,\s*"  # end_version
        r"(\w+)\s*,"  # provider
    ),
    # ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(name, domain, start_ver, end_ver, type, provider, builder, ...)
    re.compile(
        r"ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX\s*\(\s*"
        r"(\w+)\s*,\s*"  # name
        r"(\w+)\s*,\s*"  # domain
        r"(\d+)\s*,\s*"  # start_version
        r"(\d+)\s*,\s*"  # end_version
        r"(\w+)\s*,\s*"  # type
        r"(\w+)\s*,"  # provider
    ),
]

# Patterns for contrib ops (CUDA_MS_OP macros expand to ONNX_OPERATOR macros internally)
# Just match the ONNX_OPERATOR_*_KERNEL_EX patterns since the CUDA_MS_OP macros are wrappers.

# Terminal kernel registration macro names (op_name at arg 0, domain at arg 1 in all variants)
_TERMINAL_KERNEL_MACROS = {
    "ONNX_OPERATOR_KERNEL_EX",
    "ONNX_OPERATOR_TYPED_KERNEL_EX",
    "ONNX_OPERATOR_VERSIONED_KERNEL_EX",
    "ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX",
    "ONNX_OPERATOR_TWO_TYPED_KERNEL_EX",
    "ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_EX",
}


def _preprocess_content(file_path):
    """Read a file and preprocess: strip comments, join continuation lines."""
    try:
        content = Path(file_path).read_text(errors="replace")
    except OSError:
        return None
    content = re.sub(r"//.*?$", "", content, flags=re.MULTILINE)
    content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
    content = re.sub(r"\\\s*\n\s*", " ", content)
    return content


def _strip_define_bodies(content):
    """Replace #define lines with blanks so regex only matches non-define code."""
    lines = content.split("\n")
    return "\n".join("" if re.match(r"\s*#\s*define\b", line) else line for line in lines)


def _parse_macro_args_at(text, pos):
    """Parse balanced-parentheses argument list starting at *pos* (must be '(').
    Returns a list of argument strings, or None on failure."""
    if pos >= len(text) or text[pos] != "(":
        return None
    depth = 0
    args = []
    arg_start = pos + 1
    for i in range(pos, len(text)):
        ch = text[i]
        if ch == "(":
            if depth == 0:
                arg_start = i + 1
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                arg = text[arg_start:i].strip()
                if arg or args:  # keep empty trailing arg if there were prior args
                    args.append(arg)
                return args
        elif ch == "," and depth == 1:
            args.append(text[arg_start:i].strip())
            arg_start = i + 1
    return None


def _parse_macro_definitions(content):
    """Parse ``#define NAME(params) body`` statements.
    Returns ``{name: (param_name_list, body_string)}``."""
    macros = {}
    for m in re.finditer(r"^#\s*define\s+(\w+)\s*\(([^)]*)\)\s*(.*)", content, re.MULTILINE):
        name = m.group(1)
        params = [p.strip() for p in m.group(2).split(",") if p.strip() and p.strip() != "..."]
        body = m.group(3).strip()
        macros[name] = (params, body)
    return macros


def _find_calls_in_text(text, target_names):
    """Find macro invocations in *text* whose name is in *target_names*.
    Returns list of ``(macro_name, [arg, ...])``."""
    results = []
    for m in re.finditer(r"\b(\w+)\s*\(", text):
        name = m.group(1)
        if name in target_names:
            args = _parse_macro_args_at(text, m.end() - 1)
            if args is not None:
                results.append((name, args))
    return results


def _resolve_through_chain(info_field, call_args, parent_params):
    """Resolve an ``('param', idx)`` or ``('literal', val)`` through one macro-call level."""
    kind, value = info_field
    if kind == "literal":
        return info_field
    if kind == "param":
        idx = value
        if idx < len(call_args):
            arg_val = call_args[idx].strip()
            if arg_val in parent_params:
                return ("param", parent_params.index(arg_val))
            return ("literal", arg_val)
    return info_field


def _resolve_kernel_macros(macros):
    """Determine which macros transitively expand to ``ONNX_OPERATOR_*_KERNEL_EX``
    and how their parameters map to (op_name, domain).

    Returns ``{macro_name: [(op_name_info, domain_info), ...]}`` where each
    ``*_info`` is ``('literal', str_value)`` or ``('param', int_index)``.
    """
    kernel_macros = {}  # macro_name -> [(op_info, dom_info), ...]

    # Phase 1: macros whose body directly contains a terminal KERNEL_EX call
    for macro_name, (params, body) in macros.items():
        calls = _find_calls_in_text(body, _TERMINAL_KERNEL_MACROS)
        entries = set()
        for _call_name, call_args in calls:
            if len(call_args) >= 2:
                a0, a1 = call_args[0].strip(), call_args[1].strip()
                op_info = ("param", params.index(a0)) if a0 in params else ("literal", a0)
                dom_info = ("param", params.index(a1)) if a1 in params else ("literal", a1)
                entries.add((op_info, dom_info))
        if entries:
            kernel_macros[macro_name] = list(entries)

    # Phase 2: iteratively resolve higher-level wrapper macros
    changed = True
    for _ in range(20):  # bounded iteration
        if not changed:
            break
        changed = False
        for macro_name, (params, body) in macros.items():
            if macro_name in kernel_macros:
                continue
            calls = _find_calls_in_text(body, set(kernel_macros.keys()))
            entries = set()
            for call_name, call_args in calls:
                for child_op, child_dom in kernel_macros[call_name]:
                    new_op = _resolve_through_chain(child_op, call_args, params)
                    new_dom = _resolve_through_chain(child_dom, call_args, params)
                    entries.add((new_op, new_dom))
            if entries:
                kernel_macros[macro_name] = list(entries)
                changed = True

    return kernel_macros


def _extract_wrapper_registrations(content, file_path):
    """Extract registrations from wrapper macros that expand to KERNEL_EX."""
    macros = _parse_macro_definitions(content)
    if not macros:
        return []

    kernel_macros = _resolve_kernel_macros(macros)
    if not kernel_macros:
        return []

    non_define = _strip_define_bodies(content)
    invocations = _find_calls_in_text(non_define, set(kernel_macros.keys()))

    registrations = []
    seen = set()
    for call_name, call_args in invocations:
        for op_info, dom_info in kernel_macros[call_name]:
            # Resolve op_name
            if op_info[0] == "literal":
                op_name = op_info[1]
            elif op_info[0] == "param" and op_info[1] < len(call_args):
                op_name = call_args[op_info[1]].strip()
            else:
                continue

            # Resolve domain
            if dom_info[0] == "literal":
                domain = dom_info[1]
            elif dom_info[0] == "param" and dom_info[1] < len(call_args):
                domain = call_args[dom_info[1]].strip()
            else:
                domain = "kOnnxDomain"

            # Filter out C++ types / param names that aren't valid op names
            if not op_name or not op_name[0].isupper():
                continue

            key = (op_name, domain, file_path)
            if key not in seen:
                seen.add(key)
                registrations.append((op_name, domain, 0, file_path))

    return registrations


def extract_kernel_registrations(file_path):
    """Extract (op_name, domain, since_version) tuples from kernel registration macros in a file."""
    content = _preprocess_content(file_path)
    if content is None:
        return []

    registrations = []

    # Phase 1: Direct ONNX_OPERATOR_*_KERNEL_EX patterns outside #define bodies
    non_define = _strip_define_bodies(content)
    for pattern in KERNEL_EX_PATTERNS:
        for m in pattern.finditer(non_define):
            groups = m.groups()
            op_name = groups[0]
            domain = groups[1]
            since_version = int(groups[2])
            registrations.append((op_name, domain, since_version, str(file_path)))

    # Phase 2: Wrapper macros that expand to ONNX_OPERATOR_*_KERNEL_EX
    registrations.extend(_extract_wrapper_registrations(content, str(file_path)))

    return registrations


def parse_registration_table(file_path, table_func_name):
    """Parse the registration table function to extract op names referenced in BuildKernelCreateInfo calls."""
    registrations = set()
    try:
        content = Path(file_path).read_text(errors="replace")
    except OSError:
        return registrations

    # Find the function
    func_start = content.find(f"{table_func_name}")
    if func_start < 0:
        return registrations

    # Extract class names from BuildKernelCreateInfo<CLASS_NAME> entries
    # Pattern: ONNX_OPERATOR_*_KERNEL_CLASS_NAME(provider, domain, ver, [type,] name)
    class_name_patterns = [
        # ONNX_OPERATOR_KERNEL_CLASS_NAME(provider, domain, ver, name)
        re.compile(r"ONNX_OPERATOR_KERNEL_CLASS_NAME\s*\(\s*\w+\s*,\s*(\w+)\s*,\s*(\d+)\s*,\s*(\w+)\s*\)"),
        # ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(provider, domain, ver, type, name)
        re.compile(
            r"ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME\s*\(\s*\w+\s*,\s*(\w+)\s*,\s*(\d+)\s*,\s*\w+\s*,\s*(\w+)\s*\)"
        ),
        # ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(provider, domain, start, end, name)
        re.compile(
            r"ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME\s*\(\s*\w+\s*,\s*(\w+)\s*,\s*(\d+)\s*,\s*\d+\s*,\s*(\w+)\s*\)"
        ),
        # ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(provider, domain, start, end, type, name)
        re.compile(
            r"ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME\s*\(\s*\w+\s*,\s*(\w+)\s*,\s*(\d+)\s*,\s*\d+\s*,\s*\w+\s*,\s*(\w+)\s*\)"
        ),
    ]

    # Also handle CUDA_MS_OP_CLASS_NAME and CUDA_MS_OP_TYPED_CLASS_NAME
    ms_op_patterns = [
        # CUDA_MS_OP_CLASS_NAME(ver, name)  -> domain is kMSDomain
        re.compile(r"CUDA_MS_OP_CLASS_NAME\s*\(\s*(\d+)\s*,\s*(\w+)\s*\)"),
        # CUDA_MS_OP_TYPED_CLASS_NAME(ver, type, name)
        re.compile(r"CUDA_MS_OP_TYPED_CLASS_NAME\s*\(\s*(\d+)\s*,\s*\w+\s*,\s*(\w+)\s*\)"),
    ]

    # Scan from function start to end of file (conservative)
    search_region = content[func_start:]

    for pattern in class_name_patterns:
        for m in pattern.finditer(search_region):
            domain, version, name = m.group(1), int(m.group(2)), m.group(3)
            registrations.add((name, domain, version))

    for pattern in ms_op_patterns:
        for m in pattern.finditer(search_region):
            version, name = int(m.group(1)), m.group(2)
            registrations.add((name, "kMSDomain", version))

    return registrations


def get_excluded_files(cmake_path, repo_root):
    """Parse the plugin CMake file to get regex exclusion patterns."""
    exclusion_patterns = []
    try:
        content = Path(cmake_path).read_text()
    except OSError:
        return exclusion_patterns

    # Match: list(FILTER CC_SRCS EXCLUDE REGEX "pattern")
    # or:    list(FILTER CU_SRCS EXCLUDE REGEX "pattern")
    for m in re.finditer(r'list\s*\(\s*FILTER\s+\w+\s+EXCLUDE\s+REGEX\s+"([^"]+)"\s*\)', content):
        pat = m.group(1)
        # Only keep non-commented lines
        line_start = content.rfind("\n", 0, m.start()) + 1
        line = content[line_start : m.start()]
        if "#" not in line:
            exclusion_patterns.append(pat)

    return exclusion_patterns


def find_kernel_files(base_dirs, extensions=(".cc",)):
    """Find all kernel source files in the given directories."""
    files = []
    for base_dir in base_dirs:
        for ext in extensions:
            for path in Path(base_dir).rglob(f"*{ext}"):
                files.append(str(path))
    return sorted(files)


def is_excluded(file_path, exclusion_patterns):
    """Check if a file path matches any exclusion pattern."""
    return any(re.search(pat, file_path) for pat in exclusion_patterns)


def generate_report(repo_root):
    """Generate the full parity report."""
    repo_root = Path(repo_root)

    # Paths
    cuda_ep_cc = repo_root / "onnxruntime/core/providers/cuda/cuda_execution_provider.cc"
    cuda_nhwc_cc = repo_root / "onnxruntime/core/providers/cuda/cuda_nhwc_kernels.cc"
    contrib_cc = repo_root / "onnxruntime/contrib_ops/cuda/cuda_contrib_kernels.cc"
    plugin_cmake = repo_root / "cmake/onnxruntime_providers_cuda_plugin.cmake"

    # 1. Parse bundled EP registration tables
    bundled_standard = parse_registration_table(cuda_ep_cc, "RegisterCudaKernels")
    bundled_nhwc = parse_registration_table(cuda_nhwc_cc, "RegisterCudaKernels")  # NHWC uses same function name pattern
    bundled_contrib = parse_registration_table(contrib_cc, "RegisterCudaContribKernels")

    # 2. Get exclusion patterns from plugin CMake
    exclusion_patterns = get_excluded_files(plugin_cmake, repo_root)

    # 3. Scan all CUDA kernel source files
    core_cuda_dir = repo_root / "onnxruntime/core/providers/cuda"
    contrib_cuda_dir = repo_root / "onnxruntime/contrib_ops/cuda"

    all_cc_files = find_kernel_files([core_cuda_dir, contrib_cuda_dir])

    # 4. Categorize files and extract registrations
    plugin_registrations = []  # (op, domain, ver, file) tuples - compiled into plugin
    excluded_registrations = []  # (op, domain, ver, file) tuples - excluded from plugin

    for f in all_cc_files:
        regs = extract_kernel_registrations(f)
        if not regs:
            continue
        if is_excluded(f, exclusion_patterns):
            excluded_registrations.extend(regs)
        else:
            plugin_registrations.extend(regs)

    # 5. Build op sets for comparison
    plugin_ops = set()
    for op, domain, ver, _ in plugin_registrations:
        plugin_ops.add((op, domain, ver))

    excluded_ops = set()
    for op, domain, ver, _ in excluded_registrations:
        excluded_ops.add((op, domain, ver))

    # Unique op names (ignoring version/type variants)
    plugin_op_names = {(op, domain) for op, domain, _ in plugin_ops}
    excluded_op_names = {(op, domain) for op, domain, _ in excluded_ops}
    bundled_op_names = set()
    for ops_set in [bundled_standard, bundled_nhwc, bundled_contrib]:
        for op, domain, _ver in ops_set:
            bundled_op_names.add((op, domain))

    # 6. Generate report
    report = []
    report.append("=" * 70)
    report.append("CUDA EP Plugin — Kernel Registration Parity Report")
    report.append("=" * 70)
    report.append("")

    report.append("## Summary")
    report.append("  NOTE: Plugin macro counts may undercount due to nested macro")
    report.append("  expansion (e.g., BINARY_OP_VERSIONED_UZILHFD wraps multiple")
    report.append("  ONNX_OPERATOR_TYPED_KERNEL_EX calls). Bundled table counts")
    report.append("  from RegisterCudaKernels/RegisterCudaContribKernels are accurate.")
    report.append("")
    report.append("  Bundled EP registration table entries:")
    report.append(f"    Standard ops:  {len(bundled_standard)}")
    report.append(f"    NHWC ops:      {len(bundled_nhwc)}")
    report.append(f"    Contrib ops:   {len(bundled_contrib)}")
    report.append(f"    Total:         {len(bundled_standard) + len(bundled_nhwc) + len(bundled_contrib)}")
    report.append("")
    report.append("  Plugin kernel macro invocations (in compiled .cc files):")
    report.append(f"    Total:         {len(plugin_registrations)}")
    report.append("  Excluded kernel macro invocations:")
    report.append(f"    Total:         {len(excluded_registrations)}")
    report.append("")
    report.append("  Unique op names (op, domain):")
    report.append(f"    In plugin:     {len(plugin_op_names)}")
    report.append(f"    Excluded:      {len(excluded_op_names)}")
    report.append(f"    In bundled:    {len(bundled_op_names)}")
    report.append("")

    # Plugin-only ops (in plugin but not in bundled table — likely already handled)
    plugin_only = plugin_op_names - bundled_op_names
    if plugin_only:
        report.append(f"  Plugin-only op names (not in bundled table): {len(plugin_only)}")
        for op, domain in sorted(plugin_only):
            report.append(f"    - {op} ({domain})")
        report.append("")

    # Bundled-only ops (in bundled but not in plugin+excluded)
    all_source_ops = plugin_op_names | excluded_op_names
    bundled_only = bundled_op_names - all_source_ops
    if bundled_only:
        report.append(f"  Bundled-only op names (not in any .cc file KERNEL_EX): {len(bundled_only)}")
        for op, domain in sorted(bundled_only):
            report.append(f"    - {op} ({domain})")
        report.append("")

    # Coverage ratio
    if bundled_op_names:
        coverage = len(plugin_op_names & bundled_op_names) / len(bundled_op_names) * 100
        report.append(f"  Plugin coverage: {coverage:.1f}% of bundled unique op names")
    report.append("")

    # 7. Excluded ops detail
    report.append("## Excluded Ops by Category")
    report.append("")

    # Group excluded by file/directory
    excluded_by_dir = defaultdict(list)
    for op, domain, ver, filepath in excluded_registrations:
        # Extract a short category from the path
        rel_path = str(filepath).replace(str(repo_root) + "/", "")
        parts = rel_path.split("/")
        # Find the most descriptive sub-directory
        if "contrib_ops" in rel_path:
            idx = parts.index("cuda") if "cuda" in parts else 0
            category = "/".join(parts[idx + 1 : -1]) or parts[-1]
        elif "core/providers/cuda" in rel_path:
            idx = [i for i, p in enumerate(parts) if p == "cuda"][-1]
            category = "/".join(parts[idx + 1 : -1]) or parts[-1]
        else:
            category = "other"
        excluded_by_dir[category].append((op, domain, ver, rel_path))

    for category in sorted(excluded_by_dir):
        entries = excluded_by_dir[category]
        unique_ops = {(op, domain) for op, domain, _, _ in entries}
        report.append(f"  [{category}] ({len(entries)} registrations, {len(unique_ops)} unique ops)")
        for op, domain in sorted(unique_ops):
            report.append(f"    - {op} ({domain})")
        report.append("")

    report.append("## Active CMake Exclusion Patterns")
    for i, pat in enumerate(exclusion_patterns, 1):
        report.append(f"  {i:2d}. {pat}")
    report.append("")

    return "\n".join(report)


# ============================================================================
# Runtime-based parity report (uses the actual kernel registries)
# ============================================================================

# Map C++ domain constants to the string forms used by the bundled EP table parser.
_DOMAIN_CONST_TO_STRING = {
    "kOnnxDomain": "",
    "kMSDomain": "com.microsoft",
    "kMSInternalNHWCDomain": "com.microsoft.internal.nhwc",
    "kPytorchAtenDomain": "com.microsoft.pytorch.aten",
}

# Reverse: runtime domain string -> constant name for display.
_DOMAIN_STRING_TO_CONST = {v: k for k, v in _DOMAIN_CONST_TO_STRING.items()}
_DOMAIN_STRING_TO_CONST["ai.onnx"] = "kOnnxDomain"


def _runtime_domain_display(domain_str):
    """Convert a runtime domain string (e.g. '' or 'com.microsoft') to the constant name used in reports."""
    return _DOMAIN_STRING_TO_CONST.get(domain_str, domain_str or "kOnnxDomain")


def _kernel_defs_to_op_names(kernel_defs):
    """Convert a list of KernelDef objects to a set of (op_name, domain_display) tuples."""
    op_names = set()
    for kd in kernel_defs:
        domain = _runtime_domain_display(kd.domain)
        op_names.add((kd.op_name, domain))
    return op_names


def generate_runtime_report(plugin_ep_name, plugin_lib_path, bundled_ep_name="CUDAExecutionProvider"):
    """Generate a parity report by querying actual kernel registries at runtime."""
    import onnxruntime as ort  # noqa: PLC0415
    import onnxruntime.capi.onnxruntime_pybind11_state as rtpy  # noqa: PLC0415

    # 1. Get bundled EP kernel defs
    bundled_defs = [kd for kd in rtpy.get_all_opkernel_def() if kd.provider == bundled_ep_name]
    bundled_op_names = _kernel_defs_to_op_names(bundled_defs)

    # 2. Register the plugin EP
    try:
        ort.register_execution_provider_library(plugin_ep_name, plugin_lib_path)
    except Exception as e:
        if "already registered" not in str(e).lower():
            print(f"Error: failed to register plugin EP '{plugin_ep_name}': {e}", file=sys.stderr)
            sys.exit(1)

    # 3. Get plugin EP kernel defs via the C++ registry query API
    if not hasattr(rtpy, "get_registered_ep_kernel_defs"):
        raise RuntimeError(
            "get_registered_ep_kernel_defs is not available. "
            "Rebuild onnxruntime with the pybind change in onnxruntime_pybind_schema.cc."
        )

    plugin_defs = rtpy.get_registered_ep_kernel_defs(plugin_ep_name)
    plugin_op_names = _kernel_defs_to_op_names(plugin_defs)
    method = "kernel registry query"

    # 4. Build report
    report = []
    report.append("=" * 70)
    report.append("CUDA EP Plugin — Runtime Kernel Parity Report")
    report.append("=" * 70)
    report.append("")

    report.append(f"## Summary (Runtime — {method})")
    report.append(f"  Bundled EP ({bundled_ep_name}):")
    report.append(f"    Total kernel registrations:  {len(bundled_defs)}")
    report.append(f"    Unique op names (op, domain): {len(bundled_op_names)}")
    report.append("")
    report.append(f"  Plugin EP ({plugin_ep_name}):")
    if plugin_defs is not None:
        report.append(f"    Total kernel registrations:  {len(plugin_defs)}")
    report.append(f"    Unique op names (op, domain): {len(plugin_op_names)}")
    report.append("")

    plugin_only = plugin_op_names - bundled_op_names
    if plugin_only:
        report.append(f"  Plugin-only op names (not in bundled): {len(plugin_only)}")
        for op, domain in sorted(plugin_only):
            report.append(f"    - {op} ({domain})")
        report.append("")

    bundled_only = bundled_op_names - plugin_op_names
    if bundled_only:
        report.append(f"  Bundled-only op names (not in plugin): {len(bundled_only)}")
        for op, domain in sorted(bundled_only):
            report.append(f"    - {op} ({domain})")
        report.append("")

    common = bundled_op_names & plugin_op_names
    if bundled_op_names:
        coverage = len(common) / len(bundled_op_names) * 100
        report.append(
            f"  Plugin coverage: {coverage:.1f}% of bundled unique op names ({len(common)}/{len(bundled_op_names)})"
        )
    report.append("")

    # 5. Detailed version/type comparison (only with registry API)
    if plugin_defs is not None:
        bundled_by_op = defaultdict(list)
        for kd in bundled_defs:
            key = (kd.op_name, _runtime_domain_display(kd.domain))
            bundled_by_op[key].append(kd)

        plugin_by_op = defaultdict(list)
        for kd in plugin_defs:
            key = (kd.op_name, _runtime_domain_display(kd.domain))
            plugin_by_op[key].append(kd)

        version_gaps = []
        for op_key in sorted(common):
            b_versions = {kd.version_range for kd in bundled_by_op[op_key]}
            p_versions = {kd.version_range for kd in plugin_by_op[op_key]}
            missing = b_versions - p_versions
            if missing:
                version_gaps.append((op_key, missing))

        if version_gaps:
            report.append("## Version Gaps (op present but some version ranges missing in plugin)")
            for (op, domain), missing in version_gaps:
                ranges = ", ".join(f"[{v[0]}, {v[1]}]" if v[1] < 2147483647 else f"{v[0]}+" for v in sorted(missing))
                report.append(f"  {op} ({domain}): missing versions {ranges}")
            report.append("")

    return "\n".join(report)


# _probe_plugin_ops and _make_probe_model removed — session-based probing
# has been moved to test_cuda_plugin_ep.py (test_plugin_ep_claims_key_ops).
# The runtime report now requires the C++ get_registered_ep_kernel_defs API.


def main():
    parser = argparse.ArgumentParser(description="CUDA EP Plugin Registration Parity Report")
    parser.add_argument("--repo-root", default=None, help="Path to onnxruntime repo root")
    parser.add_argument(
        "--runtime",
        action="store_true",
        help="Use runtime kernel registry queries instead of static source analysis. "
        "Requires a built onnxruntime with the plugin EP library available.",
    )
    parser.add_argument(
        "--plugin-ep-name",
        default="CudaPluginExecutionProvider",
        help="Name of the plugin EP (default: CudaPluginExecutionProvider)",
    )
    parser.add_argument(
        "--plugin-ep-lib",
        default=None,
        help="Path to the plugin EP shared library. Auto-detected from build dir if not specified.",
    )
    args = parser.parse_args()

    if args.runtime:
        # Auto-detect plugin library path if not specified
        lib_path = args.plugin_ep_lib
        if lib_path is None:
            lib_path = _auto_detect_plugin_lib(args.repo_root)
        if lib_path is None:
            print(
                "Error: could not auto-detect plugin EP library. Use --plugin-ep-lib to specify the path.",
                file=sys.stderr,
            )
            sys.exit(1)

        report = generate_runtime_report(args.plugin_ep_name, lib_path)
    else:
        if args.repo_root:
            repo_root = args.repo_root
        else:
            script_dir = Path(__file__).resolve().parent
            repo_root = script_dir.parent.parent
            if not (Path(repo_root) / "onnxruntime").exists():
                print("Error: Could not find repo root. Use --repo-root flag.", file=sys.stderr)
                sys.exit(1)

        report = generate_report(repo_root)

    print(report)


def _auto_detect_plugin_lib(repo_root):
    """Try to find the plugin EP shared library in common build directories."""
    if repo_root is None:
        script_dir = Path(__file__).resolve().parent
        repo_root = script_dir.parent.parent

    repo_root = Path(repo_root)
    lib_name = "libonnxruntime_providers_cuda_plugin.so"

    # Check ORT_CUDA_PLUGIN_PATH env var first
    env_path = os.environ.get("ORT_CUDA_PLUGIN_PATH")
    if env_path and Path(env_path).exists():
        return env_path

    # Search common build directories (pick the newest if multiple exist)
    build_root = repo_root / "build"
    if build_root.is_dir():
        candidates = sorted(build_root.rglob(lib_name), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            return str(candidates[0])

    # Check onnxruntime package installation
    try:
        import onnxruntime  # noqa: PLC0415

        pkg_dir = Path(onnxruntime.__file__).parent / "capi"
        candidate = pkg_dir / lib_name
        if candidate.exists():
            return str(candidate)
    except ImportError:
        # onnxruntime is not installed in the current environment. Return None here and the script will fail later.
        pass

    return None


if __name__ == "__main__":
    main()
