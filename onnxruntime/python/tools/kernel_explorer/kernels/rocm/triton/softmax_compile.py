import triton
import triton.language as tl
import json
import shutil
import os

@triton.jit
def softmax_kernel(
    output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    # The rows of the softmax are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    # Substract maximum for numerical stability
    row_minus_max = row - tl.max(row, axis=0)
    # Note that exponentials in Triton are fast but approximate (i.e., think __expf in CUDA)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)

def compile(name_mp, lib_dir):
    def compile_one(sig, block_size):
        num_warps = 4
        if block_size >= 2048:
            num_warps = 8
        if block_size >= 4096:
            num_warps = 16

        ret = triton.compile(softmax_kernel, signature=sig, num_warps=num_warps, constants={"BLOCK_SIZE": block_size})
        return ret

    metadata = []
    for name in name_mp:
        sig, block_size = name_mp[name]
        ret = compile_one(sig, block_size)
        compile_res = {}
        compile_res['name'] = name
        compile_res['func_name'] = ret.metadata['name']
        compile_res['num_warps'] = ret.metadata['num_warps']
        compile_res['shared'] = ret.metadata['shared']
        compile_res['BLOCK_SIZE'] = block_size
        # move tmp hsaco file into current dir
        lib_name = f'{name}.hasco'
        shutil.copyfile(ret.asm['hsaco_path'], f'{lib_dir}/{lib_name}')
        compile_res['lib_file'] = lib_name
        metadata.append(compile_res)

    return metadata

dtypes = ['fp32', 'fp16']
onnx_types = ['float', 'half']
blocks = [1024, 2048, 4096, 8192, 16384]
name_pattern = 'softmax_{}_{}'
sig_pattern = '*{},*{},i32,i32,i32'

def main():
    name_mp = {}
    for dtype, onnx_t in zip(dtypes, onnx_types):
        for b in blocks:
            name = name_pattern.format(onnx_t, b)
            sig = sig_pattern.format(dtype, dtype)
            name_mp[name] = [sig, b]

    lib_dir = './libs'
    if not os.path.exists(lib_dir):
        os.mkdir(lib_dir)

    metadata = compile(name_mp, lib_dir)
    print('compile done.')

    # save metadata into json file
    with open(f'{lib_dir}/meta.json', 'w') as fp:
        json.dump(metadata, fp)
    print('save into file done.')

if __name__ == '__main__':
    main()

