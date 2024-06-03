import sys

def read_values(filename):
    with open(filename, 'r') as file:
        values = [(float(line.strip()), i + 1) for i, line in enumerate(file)]
    return values

def top_n(values, N):
    return sorted(values, key=lambda x: x[0], reverse=True)[:N]

def compare_files(cpu_file, npu_file, N):
    cpu_values = read_values(cpu_file)
    npu_values = read_values(npu_file)

    cpu_topn = top_n(cpu_values, N)
    npu_topn = top_n(npu_values, N)

    print(f"Top-{N} values in {cpu_file}: {cpu_topn}")
    print(f"Top-{N} values in {npu_file}: {npu_topn}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python compare_topn.py <N> <cpu_file> <npu_file>")
        sys.exit(1)

    N = int(sys.argv[1])
    cpu_file = sys.argv[2]
    npu_file = sys.argv[3]

    compare_files(cpu_file, npu_file, N)
