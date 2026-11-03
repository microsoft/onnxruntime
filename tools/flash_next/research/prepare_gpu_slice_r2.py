from pathlib import Path


ROOT = Path(__file__).parent


if __name__ == "__main__":
    for original, target in (
        ("run_gpu_slice_toy.py", "run_gpu_slice_toy_r2.py"),
        ("run_gpu_slice_wide.py", "run_gpu_slice_wide_r2.py"),
    ):
        source = (ROOT / original).read_text().replace("v6-r1", "v6-r2")
        source = source.replace('"run_gpu_slice_wide.py"', '"run_gpu_slice_wide_r2.py"')
        with (ROOT / target).open("x") as stream:
            stream.write(source)
