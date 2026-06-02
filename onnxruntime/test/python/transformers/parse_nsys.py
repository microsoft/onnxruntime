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

    # Parse kernel timings
    patterns = args.pattern if args.pattern else None
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
