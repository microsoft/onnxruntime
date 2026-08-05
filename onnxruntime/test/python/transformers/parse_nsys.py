#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""
Parse nsys SQLite output to extract CUDA kernel timings.

Usage:
  # First, profile with nsys:
  nsys profile -o sln_fp16 --export=sqlite python profile_skip_layer_norm.py --mode fp16 --warmup 5 --repeat 100
  nsys profile -o gqa_int8 --export=sqlite python profile_gqa.py --mode int8 --warmup 5 --repeat 10

  # Then parse the results (using NVTX marker to exclude warmup):
  python parse_nsys.py sln_fp16.sqlite --nvtx-range benchmark
  python parse_nsys.py gqa_int8.sqlite --nvtx-range benchmark --output results.json
  python parse_nsys.py gqa_int8.sqlite --format csv --output results.csv

  # Alternative: skip first N calls per kernel to exclude warmup:
  python parse_nsys.py sln_fp16.sqlite --skip-first 5
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path


def parse_nsys_sqlite(
    db_path: str,
    kernel_patterns: list[str] | None = None,
    skip_first: int = 0,
    nvtx_range: str | None = None,
) -> list[dict]:
    """
    Parse nsys SQLite database and extract kernel timing information.

    Args:
        db_path: Path to the .sqlite file exported by nsys
        kernel_patterns: List of SQL LIKE patterns to filter kernels (default: onnxruntime patterns)
        skip_first: Number of initial kernel calls to skip per kernel type (e.g., to exclude warmup)
        nvtx_range: If specified, only include kernels launched within this NVTX range

    Returns:
        List of dicts with kernel timing info
    """
    if kernel_patterns is None:
        kernel_patterns = [
            "%onnxruntime%",
        ]

    conn = sqlite3.connect(db_path)

    # Build WHERE clause for kernel patterns using parameterized queries
    pattern_placeholders = " OR ".join(["s.value LIKE ?" for _ in kernel_patterns])
    params: list = list(kernel_patterns)

    if nvtx_range:
        # Filter kernels that launched within the specified NVTX range
        if skip_first > 0:
            query = f"""
            WITH numbered AS (
                SELECT
                    s.value as kernel_name,
                    k.end - k.start as duration_ns,
                    ROW_NUMBER() OVER (PARTITION BY s.value ORDER BY k.start) as call_num
                FROM CUPTI_ACTIVITY_KIND_KERNEL k
                JOIN StringIds s ON k.demangledName = s.id
                JOIN NVTX_EVENTS n ON k.start >= n.start AND k.start <= n.end
                JOIN StringIds ns ON n.textId = ns.id
                WHERE ({pattern_placeholders}) AND ns.value = ?
            )
            SELECT
                kernel_name,
                SUM(duration_ns) as total_ns,
                COUNT(*) as call_count,
                MIN(duration_ns) as min_ns,
                MAX(duration_ns) as max_ns,
                AVG(duration_ns) as avg_ns
            FROM numbered
            WHERE call_num > ?
            GROUP BY kernel_name
            ORDER BY total_ns DESC
            """
            params.append(nvtx_range)
            params.append(skip_first)
        else:
            query = f"""
            SELECT
                s.value as kernel_name,
                SUM(k.end - k.start) as total_ns,
                COUNT(*) as call_count,
                MIN(k.end - k.start) as min_ns,
                MAX(k.end - k.start) as max_ns,
                AVG(k.end - k.start) as avg_ns
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON k.demangledName = s.id
            JOIN NVTX_EVENTS n ON k.start >= n.start AND k.start <= n.end
            JOIN StringIds ns ON n.textId = ns.id
            WHERE ({pattern_placeholders}) AND ns.value = ?
            GROUP BY s.value
            ORDER BY total_ns DESC
            """
            params.append(nvtx_range)
    elif skip_first > 0:
        # Use window function to number calls and skip first N per kernel type
        query = f"""
        WITH numbered AS (
            SELECT
                s.value as kernel_name,
                k.end - k.start as duration_ns,
                ROW_NUMBER() OVER (PARTITION BY s.value ORDER BY k.start) as call_num
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON k.demangledName = s.id
            WHERE {pattern_placeholders}
        )
        SELECT
            kernel_name,
            SUM(duration_ns) as total_ns,
            COUNT(*) as call_count,
            MIN(duration_ns) as min_ns,
            MAX(duration_ns) as max_ns,
            AVG(duration_ns) as avg_ns
        FROM numbered
        WHERE call_num > ?
        GROUP BY kernel_name
        ORDER BY total_ns DESC
        """
        params.append(skip_first)
    else:
        # Original query without skipping
        query = f"""
        SELECT
            s.value as kernel_name,
            SUM(k.end - k.start) as total_ns,
            COUNT(*) as call_count,
            MIN(k.end - k.start) as min_ns,
            MAX(k.end - k.start) as max_ns,
            AVG(k.end - k.start) as avg_ns
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.demangledName = s.id
        WHERE {pattern_placeholders}
        GROUP BY s.value
        ORDER BY total_ns DESC
        """

    results = []
    try:
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        for row in rows:
            results.append(
                {
                    "kernel_name": row[0],
                    "total_ms": row[1] / 1e6,  # ns to ms
                    "call_count": row[2],
                    "min_us": row[3] / 1e3,  # ns to us
                    "max_us": row[4] / 1e3,
                    "avg_us": row[5] / 1e3,
                }
            )
    except sqlite3.OperationalError as e:
        print(f"SQL Error: {e}", file=sys.stderr)

    conn.close()
    return results


def list_all_kernels(db_path: str) -> list[str]:
    """List all kernel names in the database for debugging."""
    conn = sqlite3.connect(db_path)

    try:
        # Join with StringIds to get actual kernel names
        cursor = conn.execute("""
            SELECT DISTINCT s.value
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON k.demangledName = s.id
            ORDER BY s.value
        """)
        return [row[0] for row in cursor.fetchall()]
    except sqlite3.OperationalError as e:
        print(f"SQL Error: {e}", file=sys.stderr)
        return []
    finally:
        conn.close()


# CUDA runtime API calls that block the host until the device/stream finishes
# (host-side synchronization). These should NOT appear inside a run when the
# run is launched with disable_synchronize_execution_providers=1 (sync=false).
# Patterns use SQL LIKE syntax (matched against the runtime API name, which may
# carry a version suffix, e.g. "cudaStreamSynchronize_v3000" or "cudaMemcpy_v3020").
# Only CUPTI_ACTIVITY_KIND_RUNTIME (the cuda* runtime API) is scanned, so driver
# (cu*) names are intentionally not listed here. cudaStreamWaitEvent is excluded
# because it is a stream-ordering primitive, not a host-blocking sync.
SYNC_API_PATTERNS = [
    "cudaDeviceSynchronize%",
    "cudaStreamSynchronize%",
    "cudaEventSynchronize%",
    "cudaMemcpy%",  # synchronous cudaMemcpy* (async variants excluded below)
    "cudaMemset%",  # synchronous cudaMemset* (async variants excluded below)
]

# API names matching any of these are excluded even if they match a SYNC pattern.
# This removes non-blocking *Async* copies/sets (e.g. cudaMemcpyAsync) which do
# not synchronize the host.
SYNC_API_EXCLUDE_PATTERNS = [
    "%Async%",
]


def parse_cuda_api_in_range(
    db_path: str,
    nvtx_range: str,
    api_patterns: list[str] | None = None,
    skip_first_ranges: int = 0,
    exclude_patterns: list[str] | None = None,
) -> list[dict]:
    """
    Aggregate CUDA runtime API calls that occur within an NVTX range.

    Each occurrence of the NVTX range (e.g. one per inference run) is treated as
    a separate window. CUDA runtime API events (CUPTI_ACTIVITY_KIND_RUNTIME) are
    attributed to a window when their host-side start timestamp falls inside it.

    Args:
        db_path: Path to the .sqlite file exported by nsys.
        nvtx_range: NVTX range/marker name to scope the API calls to.
        api_patterns: Optional list of SQL LIKE patterns to filter API names.
                      If None, all API calls in the range are returned.
        skip_first_ranges: Skip API calls in the first N occurrences of the NVTX
                           range (e.g. to exclude warmup iterations).
        exclude_patterns: Optional list of SQL LIKE patterns; API names matching
                          any of them are excluded (e.g. "%Async%").

    Returns:
        List of dicts with API call aggregation, ordered by call_count desc.
    """
    conn = sqlite3.connect(db_path)

    where_clauses = ["ns.value = ?"]
    params: list = [nvtx_range]

    if api_patterns:
        pattern_or = " OR ".join(["s.value LIKE ? ESCAPE '\\'" for _ in api_patterns])
        where_clauses.append(f"({pattern_or})")
        params.extend(api_patterns)

    if exclude_patterns:
        exclude_and = " AND ".join(["s.value NOT LIKE ? ESCAPE '\\'" for _ in exclude_patterns])
        where_clauses.append(f"({exclude_and})")
        params.extend(exclude_patterns)

    where_sql = " AND ".join(where_clauses)

    # Number the NVTX range occurrences so we can skip warmup windows, then attach
    # each CUDA runtime API event to the window whose [start, end] contains it.
    query = f"""
    WITH ranges AS (
        SELECT
            n.start AS r_start,
            n.end AS r_end,
            ROW_NUMBER() OVER (ORDER BY n.start) AS range_num
        FROM NVTX_EVENTS n
        JOIN StringIds ns ON n.textId = ns.id
        WHERE ns.value = ?
    )
    SELECT
        s.value AS api_name,
        COUNT(*) AS call_count,
        SUM(r.end - r.start) AS total_ns,
        MIN(r.end - r.start) AS min_ns,
        MAX(r.end - r.start) AS max_ns,
        AVG(r.end - r.start) AS avg_ns
    FROM CUPTI_ACTIVITY_KIND_RUNTIME r
    JOIN StringIds s ON r.nameId = s.id
    JOIN ranges rg ON r.start >= rg.r_start AND r.start <= rg.r_end
    JOIN NVTX_EVENTS n ON r.start >= n.start AND r.start <= n.end
    JOIN StringIds ns ON n.textId = ns.id
    WHERE {where_sql} AND rg.range_num > ?
    GROUP BY s.value
    ORDER BY call_count DESC
    """
    # ranges CTE consumes the first nvtx_range param; rebuild the full param list.
    full_params = [nvtx_range, *params, skip_first_ranges]

    results = []
    try:
        cursor = conn.execute(query, full_params)
        for row in cursor.fetchall():
            results.append(
                {
                    "api_name": row[0],
                    "call_count": row[1],
                    "total_us": row[2] / 1e3,  # ns to us
                    "min_us": row[3] / 1e3,
                    "max_us": row[4] / 1e3,
                    "avg_us": row[5] / 1e3,
                }
            )
    except sqlite3.OperationalError as e:
        print(f"SQL Error: {e}", file=sys.stderr)

    conn.close()
    return results


def list_all_cuda_apis(db_path: str) -> list[str]:
    """List all CUDA runtime API names in the database for debugging."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute("""
            SELECT DISTINCT s.value
            FROM CUPTI_ACTIVITY_KIND_RUNTIME r
            JOIN StringIds s ON r.nameId = s.id
            ORDER BY s.value
        """)
        return [row[0] for row in cursor.fetchall()]
    except sqlite3.OperationalError as e:
        print(f"SQL Error: {e}", file=sys.stderr)
        return []
    finally:
        conn.close()


def format_cuda_api_table(results: list[dict], prefix: str = "") -> str:
    """Format CUDA API aggregation results as a human-readable table."""
    if not results:
        return "No matching CUDA API calls found."

    api_name_len_limit = 48
    lines = []
    lines.append(
        f"{prefix}{'CUDA API':<{api_name_len_limit}} {'Calls':>8} {'Total(us)':>10} "
        f"{'Avg(us)':>10} {'Min(us)':>10} {'Max(us)':>10}"
    )
    lines.append("-" * 100)
    for r in results:
        name = r["api_name"]
        name = name[: api_name_len_limit - 3] + "..." if len(name) > api_name_len_limit else name
        lines.append(
            f"{name:<{api_name_len_limit}} {r['call_count']:>8d} {r['total_us']:>10.2f} "
            f"{r['avg_us']:>10.2f} {r['min_us']:>10.2f} {r['max_us']:>10.2f}"
        )
    return "\n".join(lines)


def format_kernel_name(kernel_name: str) -> str:
    prefix_list = [
        "void onnxruntime::contrib::cuda::",
        "void onnxruntime::",
        "onnxruntime::contrib::cuda::",
        "onnxruntime::",
    ]
    for prefix in prefix_list:
        if kernel_name.startswith(prefix):
            return kernel_name[len(prefix) :]
    return kernel_name


def format_table(results: list[dict], prefix: str) -> str:
    """Format results as a human-readable table."""
    if not results:
        return "No matching kernels found."

    kernel_name_len_limit = 64
    lines = []
    lines.append(
        f"{prefix}{'Kernel Name':<{kernel_name_len_limit}} {'Total(ms)':>10} {'Calls':>8} {'Avg(us)':>10} {'Min(us)':>10} {'Max(us)':>10}"
    )
    lines.append("-" * 120)

    for r in results:
        kernel_name = format_kernel_name(r["kernel_name"])
        name = (
            kernel_name[:kernel_name_len_limit] + "..." if len(kernel_name) > kernel_name_len_limit - 3 else kernel_name
        )
        lines.append(
            f"{name:<{kernel_name_len_limit}} {r['total_ms']:>10.3f} {r['call_count']:>8d} {r['avg_us']:>10.2f} {r['min_us']:>10.2f} {r['max_us']:>10.2f}"
        )

    return "\n".join(lines)


def format_csv(results: list[dict]) -> str:
    """Format results as CSV."""
    lines = ["kernel_name,total_ms,call_count,avg_us,min_us,max_us"]
    for r in results:
        lines.append(
            f'"{r["kernel_name"]}",{r["total_ms"]:.6f},{r["call_count"]},{r["avg_us"]:.3f},{r["min_us"]:.3f},{r["max_us"]:.3f}'
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Parse nsys SQLite output for CUDA kernel timings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile and parse (using NVTX range to exclude warmup):
  nsys profile -o sln --export=sqlite python profile_skip_layer_norm.py --warmup 5 --repeat 100
  python parse_nsys.py sln.sqlite --nvtx-range benchmark

  # Alternative: skip first N warmup calls per kernel:
  python parse_nsys.py sln.sqlite --skip-first 5

  # Export to JSON:
  python parse_nsys.py sln.sqlite --nvtx-range benchmark --format json --output results.json

  # List all kernels (for debugging):
  python parse_nsys.py sln.sqlite --list-kernels
        """,
    )
    parser.add_argument("sqlite_file", help="Path to nsys SQLite export file")
    parser.add_argument(
        "--format", choices=["table", "json", "csv"], default="table", help="Output format (default: table)"
    )
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")
    parser.add_argument("--list-kernels", action="store_true", help="List all kernel names in the database")
    parser.add_argument("--pattern", action="append", help="Add custom kernel name pattern (SQL LIKE syntax)")
    parser.add_argument("--tag", default="", help="Tag for kernel name in output table. Example tag: 'fp16' or 'int8'")
    parser.add_argument(
        "--nvtx-range",
        metavar="NAME",
        help="Only include kernels launched within this NVTX range (e.g., 'benchmark')",
    )
    parser.add_argument(
        "--skip-first",
        type=int,
        default=0,
        metavar="N",
        help="Skip first N calls per kernel type (e.g., to exclude warmup iterations)",
    )
    parser.add_argument(
        "--cuda-api",
        action="store_true",
        help="Aggregate CUDA runtime API calls within --nvtx-range instead of kernels",
    )
    parser.add_argument(
        "--sync-apis-only",
        action="store_true",
        help="With --cuda-api, only report host synchronization APIs (cudaStreamSynchronize, "
        "cudaDeviceSynchronize, etc.). Exit code is 1 if any are found in the range.",
    )
    parser.add_argument(
        "--api-pattern",
        action="append",
        help="With --cuda-api, add a custom CUDA API name pattern (SQL LIKE syntax)",
    )
    parser.add_argument("--list-cuda-apis", action="store_true", help="List all CUDA runtime API names in the database")
    parser.add_argument(
        "--skip-first-ranges",
        type=int,
        default=0,
        metavar="N",
        help="With --cuda-api, skip API calls in the first N occurrences of the NVTX range (warmup)",
    )

    args = parser.parse_args()

    if not Path(args.sqlite_file).exists():
        print(f"Error: File not found: {args.sqlite_file}", file=sys.stderr)
        sys.exit(1)

    if args.list_kernels:
        kernels = list_all_kernels(args.sqlite_file)
        print(f"Found {len(kernels)} unique kernels:")
        for k in kernels:
            print(f"  {k}")
        return

    if args.list_cuda_apis:
        apis = list_all_cuda_apis(args.sqlite_file)
        print(f"Found {len(apis)} unique CUDA runtime APIs:")
        for a in apis:
            print(f"  {a}")
        return

    if args.cuda_api:
        if not args.nvtx_range:
            print("Error: --cuda-api requires --nvtx-range", file=sys.stderr)
            sys.exit(2)

        if args.sync_apis_only:
            api_patterns = SYNC_API_PATTERNS
        elif args.api_pattern:
            api_patterns = args.api_pattern
        else:
            api_patterns = None

        api_results = parse_cuda_api_in_range(
            args.sqlite_file,
            args.nvtx_range,
            api_patterns=api_patterns,
            skip_first_ranges=args.skip_first_ranges,
            exclude_patterns=SYNC_API_EXCLUDE_PATTERNS if args.sync_apis_only else None,
        )

        scope = f"NVTX range '{args.nvtx_range}'"
        if args.skip_first_ranges > 0:
            scope += f" (skipping first {args.skip_first_ranges} occurrences)"

        if args.format == "json":
            output = json.dumps(api_results, indent=2)
        elif args.format == "csv":
            header = "api_name,call_count,total_us,avg_us,min_us,max_us"
            rows = [
                f'"{r["api_name"]}",{r["call_count"]},{r["total_us"]:.3f},'
                f"{r['avg_us']:.3f},{r['min_us']:.3f},{r['max_us']:.3f}"
                for r in api_results
            ]
            output = "\n".join([header, *rows])
        else:
            output = format_cuda_api_table(api_results, args.tag + " " if args.tag else "")

        if args.sync_apis_only:
            print(f"(Checking for host synchronization CUDA APIs within {scope})\n")
        else:
            print(f"(CUDA runtime API calls within {scope})\n")

        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            print(f"Results written to {args.output}")
        else:
            print(output)

        if args.sync_apis_only:
            total_sync = sum(r["call_count"] for r in api_results)
            if total_sync > 0:
                print(
                    f"\nFAIL: found {total_sync} host synchronization CUDA API call(s) within {scope}.",
                    file=sys.stderr,
                )
                sys.exit(1)
            print(f"\nPASS: no host synchronization CUDA API calls within {scope}.")
        return

    # Parse kernel timings
    patterns = args.pattern or None
    results = parse_nsys_sqlite(args.sqlite_file, patterns, skip_first=args.skip_first, nvtx_range=args.nvtx_range)

    if args.nvtx_range:
        print(f"(Filtering kernels within NVTX range: '{args.nvtx_range}')\n")
    elif args.skip_first > 0:
        print(f"(Skipping first {args.skip_first} calls per kernel)\n")

    # Format output
    if args.format == "json":
        output = json.dumps(results, indent=2)
    elif args.format == "csv":
        output = format_csv(results)
    else:
        output = format_table(results, args.tag + " " if args.tag else "")

    # Write output
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Results written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
