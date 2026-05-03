#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Build the Microsoft.ML.OnnxRuntime.EP.WebGpu NuGet package.

Stages native binaries from build artifacts into the runtimes/ layout expected
by the .csproj and runs `dotnet pack` to produce the .nupkg / .snupkg files.

Can be invoked locally or from CI. In CI, pass --artifacts-dir to point at the
downloaded pipeline artifacts. Locally, pass individual --binary-dir-* options
or place binaries manually in the runtimes/ folders.

Examples
--------
Local: pack win-x64 only from a local build:

    python pack_nuget.py --version 0.1.0-dev \\
        --binary-dir-win-x64 ../../build/webgpu.plugin/Release/Release

CI: pack all platforms from downloaded artifacts:

    python pack_nuget.py --version $(PluginPackageVersion) \\
        --artifacts-dir $(Build.BinariesDirectory)/artifacts \\
        --output-dir $(Build.ArtifactStagingDirectory)/nuget
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Platform name -> (RID, list of native binary filenames expected in the source dir).
PLATFORMS: dict[str, tuple[str, tuple[str, ...]]] = {
    "win_x64": ("win-x64", ("onnxruntime_providers_webgpu.dll", "dxil.dll", "dxcompiler.dll")),
    "win_arm64": ("win-arm64", ("onnxruntime_providers_webgpu.dll", "dxil.dll", "dxcompiler.dll")),
    "linux_x64": ("linux-x64", ("libonnxruntime_providers_webgpu.so",)),
    "macos_arm64": ("osx-arm64", ("libonnxruntime_providers_webgpu.dylib",)),
}

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR / "Microsoft.ML.OnnxRuntime.EP.WebGpu"
CSPROJ = PROJECT_DIR / "Microsoft.ML.OnnxRuntime.EP.WebGpu.csproj"
MIN_ORT_VERSION_FILE = SCRIPT_DIR.parent / "MIN_ONNXRUNTIME_VERSION"


class PackError(RuntimeError):
    """Raised for any user-actionable failure during packaging."""


def parse_args() -> argparse.Namespace:
    def _absolute_path(value: str) -> Path:
        """argparse `type` converter: parse a string as an absolute Path."""
        return Path(value).resolve()

    p = argparse.ArgumentParser(
        description="Build the Microsoft.ML.OnnxRuntime.EP.WebGpu NuGet package.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", required=True, help="Package version (e.g. 0.1.0-dev).")
    p.add_argument(
        "--output-dir",
        type=_absolute_path,
        default=(SCRIPT_DIR / "nuget_output").resolve(),
        help="Directory for the .nupkg / .snupkg output (default: ./nuget_output).",
    )
    p.add_argument("--configuration", default="Release", help="Build configuration (default: Release).")

    # CI mode: a single root containing per-platform subdirectories.
    p.add_argument(
        "--artifacts-dir",
        type=_absolute_path,
        help="CI mode: root containing <platform>/bin/ subdirectories for each platform.",
    )

    # Local mode: explicit per-platform binary directories. Each takes precedence over
    # --artifacts-dir for that platform.
    for name in PLATFORMS:
        flag = f"--binary-dir-{name.replace('_', '-')}"
        p.add_argument(flag, type=_absolute_path, dest=f"binary_dir_{name}", help=f"Path to {name} native binaries.")

    p.add_argument(
        "--nuget-config", type=_absolute_path, help="Optional NuGet.config passed to dotnet via --configfile."
    )
    p.add_argument(
        "--staging-dir",
        type=_absolute_path,
        help=(
            "Explicit staging directory. Required with --build-only / --pack-only "
            "(caller owns its lifecycle). When omitted, an auto-cleaned temporary "
            "directory is used for the full build+pack flow."
        ),
    )

    phase = p.add_mutually_exclusive_group()
    phase.add_argument(
        "--build-only",
        action="store_true",
        help="Stage and build the managed DLL only; skip dotnet pack. Preserves the staging dir.",
    )
    phase.add_argument(
        "--pack-only",
        action="store_true",
        help="Skip staging/build and run dotnet pack against an existing staging directory.",
    )

    p.add_argument(
        "--required-platforms",
        default="",
        help=(
            "Comma-separated list of platforms that MUST be staged successfully. "
            "When omitted, the script just requires at least one platform to be staged."
        ),
    )

    return p.parse_args()


def parse_required_platforms(value: str) -> list[str]:
    names = [tok.strip() for tok in value.split(",") if tok.strip()]
    invalid = [n for n in names if n not in PLATFORMS]
    if invalid:
        raise PackError(
            f"unknown platform(s) in --required-platforms: {', '.join(invalid)}. valid: {', '.join(PLATFORMS)}."
        )
    return names


def stage_sources(staging_dir: Path) -> None:
    """Copy project sources into staging, excluding bin/obj."""
    print(f"Staging project files to {staging_dir}")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    shutil.copytree(
        PROJECT_DIR,
        staging_dir,
        ignore=shutil.ignore_patterns("bin", "obj"),
    )


def resolve_platform_source(
    name: str,
    binary_dir_override: Path | None,
    artifacts_dir: Path | None,
    is_required: bool,
) -> Path | None:
    """Return the source dir for a platform, or None to skip."""
    if binary_dir_override is not None:
        return binary_dir_override
    if artifacts_dir is not None:
        candidate = artifacts_dir / name / "bin"
        if candidate.is_dir():
            return candidate
        if is_required:
            raise PackError(f"required platform '{name}' artifact directory not found: {candidate}")
    if is_required:
        raise PackError(
            f"required platform '{name}' has no binary directory "
            f"(pass --binary-dir-{name.replace('_', '-')} or --artifacts-dir)."
        )
    return None


def stage_binaries(
    staging_dir: Path,
    args: argparse.Namespace,
    required_platforms: list[str],
) -> None:
    staged: set[str] = set()

    for name, (rid, files) in PLATFORMS.items():
        binary_dir_override: Path | None = getattr(args, f"binary_dir_{name}")
        is_required = name in required_platforms
        source_dir = resolve_platform_source(name, binary_dir_override, args.artifacts_dir, is_required)
        if source_dir is None:
            print(f"Skipping {name} (no binary directory provided)")
            continue
        if not source_dir.is_dir():
            raise PackError(f"binary directory does not exist: {source_dir}")

        target_dir = staging_dir / "runtimes" / rid / "native"
        target_dir.mkdir(parents=True, exist_ok=True)

        print(f"Staging {name} -> runtimes/{rid}/native/")
        for filename in files:
            src = source_dir / filename
            if not src.is_file():
                raise PackError(f"expected binary not found: {src}")
            shutil.copy2(src, target_dir / filename)
            print(f"  {filename}")
        staged.add(name)

    if required_platforms:
        missing = [n for n in required_platforms if n not in staged]
        if missing:
            raise PackError(f"required platforms not staged: {', '.join(missing)}")
    elif not staged:
        raise PackError("no platform binaries were staged. Provide at least one --binary-dir-* or --artifacts-dir.")

    print()
    print("Runtimes layout:")
    for path in sorted((staging_dir / "runtimes").rglob("*")):
        print(f"  {path}")


def dotnet_common_args(
    staged_csproj: Path,
    args: argparse.Namespace,
    min_ort_version_file: Path,
) -> list[str]:
    common = [
        str(staged_csproj),
        "--configuration",
        args.configuration,
        f"-p:Version={args.version}",
        f"-p:OnnxRuntimeMinVersionFile={min_ort_version_file}",
    ]
    if args.nuget_config:
        common.extend(["--configfile", str(args.nuget_config)])
        print(f"Using NuGet.config: {args.nuget_config}")
    return common


def do_build(staged_csproj: Path, staging_dir: Path, args: argparse.Namespace, min_ort_version_file: Path) -> None:
    print()
    print(f"Running dotnet build (Version={args.version}, Configuration={args.configuration})...")
    cmd = ["dotnet", "build", *dotnet_common_args(staged_csproj, args, min_ort_version_file)]
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)

    # Note: "netstandard2.0" must match <TargetFramework> in Microsoft.ML.OnnxRuntime.EP.WebGpu.csproj.
    managed_dll = staging_dir / "bin" / args.configuration / "netstandard2.0" / "Microsoft.ML.OnnxRuntime.EP.WebGpu.dll"
    if not managed_dll.is_file():
        raise PackError(f"managed DLL not found after build: {managed_dll}")
    print()
    print(f"Built managed DLL: {managed_dll}")
    print("Staging directory preserved for subsequent --pack-only invocation.")


def do_pack(
    staged_csproj: Path,
    output_dir: Path,
    args: argparse.Namespace,
    min_ort_version_file: Path,
) -> None:
    print()
    print(f"Running dotnet pack (Version={args.version}, Configuration={args.configuration})...")
    pack_args = [
        "dotnet",
        "pack",
        *dotnet_common_args(staged_csproj, args, min_ort_version_file),
        "--output",
        str(output_dir),
    ]
    if args.pack_only:
        pack_args.append("--no-build")
    print("+ " + " ".join(pack_args))
    subprocess.run(pack_args, check=True)

    print()
    nupkgs = sorted(output_dir.glob("*.nupkg"))
    if not nupkgs:
        raise PackError(f"no .nupkg files found in {output_dir}")
    for pkg in nupkgs:
        print(f"Produced: {pkg.name} ({pkg.stat().st_size / (1024 * 1024):.2f} MB)")
    for pkg in sorted(output_dir.glob("*.snupkg")):
        print(f"Produced: {pkg.name} ({pkg.stat().st_size / (1024 * 1024):.2f} MB)")


def run_in_staging(args: argparse.Namespace, staging_dir: Path, min_ort_version_file: Path) -> None:
    staged_csproj = staging_dir / "Microsoft.ML.OnnxRuntime.EP.WebGpu.csproj"
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    required_platforms = parse_required_platforms(args.required_platforms)

    if args.pack_only:
        if not staged_csproj.is_file():
            raise PackError(f"staged project not found at {staged_csproj}. Run with --build-only first.")
        print(f"Reusing existing staging directory: {staging_dir}")
    else:
        stage_sources(staging_dir)
        stage_binaries(staging_dir, args, required_platforms)

    if args.build_only:
        do_build(staged_csproj, staging_dir, args, min_ort_version_file)
        return

    do_pack(staged_csproj, output_dir, args, min_ort_version_file)

    print()
    print(f"Done. Output: {output_dir}")


def run(args: argparse.Namespace) -> None:
    if not CSPROJ.is_file():
        raise PackError(f"project file not found: {CSPROJ}")
    if not MIN_ORT_VERSION_FILE.is_file():
        raise PackError(f"MIN_ONNXRUNTIME_VERSION file not found: {MIN_ORT_VERSION_FILE}")
    if args.nuget_config and not args.nuget_config.is_file():
        raise PackError(f"NuGet.config not found: {args.nuget_config}")

    if (args.build_only or args.pack_only) and not args.staging_dir:
        raise PackError("--staging-dir is required when using --build-only or --pack-only.")

    min_ort_version_file = MIN_ORT_VERSION_FILE.resolve()

    if args.staging_dir:
        staging_dir: Path = args.staging_dir
        staging_dir.mkdir(parents=True, exist_ok=True)
        run_in_staging(args, staging_dir, min_ort_version_file)
        return

    # Full build+pack flow with no caller-managed staging dir: use a temp dir that
    # is cleaned up automatically (including on exception).
    with tempfile.TemporaryDirectory(prefix="webgpu_pack_") as tmp:
        run_in_staging(args, Path(tmp), min_ort_version_file)


def main() -> int:
    args = parse_args()
    try:
        run(args)
    except PackError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as e:
        cmd_name = e.cmd[0] if e.cmd else "subprocess"
        print(f"error: {cmd_name} failed with exit code {e.returncode}", file=sys.stderr)
        return e.returncode or 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
