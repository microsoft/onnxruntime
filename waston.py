import argparse
import csv
import sys
import json

parser = argparse.ArgumentParser()
parser.add_argument('--ort_file', type=str)
parser.add_argument('--pt_file', type=str)
args = parser.parse_args()

def get_detailed_lines(run_name, path):
    lines = {}
    total_kernel_time = 0
    min_start_time = sys.maxsize
    max_end_time = 0
    with open(path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            name = str(row['Name'])
            if name not in lines:
                lines[name] = {}
            
            start_time = int(row['Start Time(ns)'])
            kernel_time = int(row['Duration(ns)'])
            end_time = start_time + kernel_time

            total_kernel_time += kernel_time
            min_start_time = min(min_start_time, start_time)
            max_end_time = max(max_end_time, end_time)

            if "Invocations" not in lines[name]:
                lines[name]["Invocations"] = 0
            lines[name]["Invocations"] += 1

            if "KernelTime" not in lines[name]:
                lines[name]["KernelTime"] = 0
            lines[name]["KernelTime"] += kernel_time
        
        for value in lines.values():
            value["Percentage"] = value["KernelTime"] / total_kernel_time * 100
    
    total_run_time = max_end_time - min_start_time
    
    print("{} Total_run_time (ns): {}".format(run_name, total_run_time))
    print("{} Total_kernel_time (ns): {}".format(run_name, total_kernel_time))
    print("{} Compute Utilization: {:.2f}%".format(run_name, total_kernel_time / total_run_time * 100))

    # lines = {key: value for key, value in lines.items() if value["Percentage"] >= 0.01}

    return lines

activities = [
    ('nccl', lambda x : x.find('nccl') >= 0),
    ('gemm', lambda x : 'gemm' in x),
    ('gemv', lambda x : any(substr in x for substr in ['cublasGemvTensorStridedBatched'])),
    ('memcpy DtoD', lambda x : any(substr in x.lower() for substr in ['memcpy dtod'])),
    ('memcpy HtoD', lambda x : any(substr in x.lower() for substr in ['memcpy htod'])),
    ('memcpy DtoH', lambda x : any(substr in x.lower() for substr in ['memcpy dtoh'])),
    ('memset', lambda x : any(substr in x for substr in ['memset'])),
    ('adam', lambda x : x.lower().find('adam') >= 0),
    ('lamb', lambda x : x.lower().find('lamb') >= 0),
    ('dropout', lambda x : x.lower().find('dropout') >= 0),
    ('gelu', lambda x : any(substr in x.lower() for substr in ['gelu'])),
    ('relu', lambda x : any(substr in x for substr in ['OP_Relu', 'threshold_kernel_impl', 'clamp_min_scalar_kernel_impl'])),
    ('sqrt', lambda x : 'sqrt' in x),
    ('non-zero', lambda x : any(substr in x.lower() for substr in ['NonZero', 'write_indices'])),
    ('compare', lambda x : any(substr in x for substr in ['CompareGEFunctor', 'CompareNEFunctor','CompareEqFunctor', 'CompareLTFunctor', 'CompareGTFunctor', 'OP_Greater', 'OP_GreaterOrEqual', 'OP_Equal', 'OP_Less', 'OP_LessOrEqual'])),
    ('isfinite', lambda x : 'foreach_non_finite_check_and_unscale' in x),
    ('layernorm', lambda x : any(substr in x for substr in ['LayerNorm', 'layer_norm', 'layernorm', 'bn_fw_tr_1C11_singleread', 'cuCompute'])),
    ('reduce', lambda x : any(substr in x for substr in ['reduce', 'op_tensor_kernel_alpha2_zero'])),
    ('cross_entropy', lambda x : any(substr in x for substr in ['XEntropy', 'CrossEntropy', 'NLLCriterion'])),
    ('softmax', lambda x : any(substr in x.lower() for substr in ['softmax'])),
    ('embedding', lambda x : any(substr in x for substr in ['indexSelectLargeIndex', 'Embedding', 'compute_grad_weight', 'krn_partials_per_segment', 'krn_partial_segment_offset'])),
    ('gather-scatter', lambda x : any(substr in x.lower() for substr in ['gather', 'scatter', 'index_kernel_impl', 'index_put_kernel_impl', 'indexing_backward_kernel'])),
    ('l2norm', lambda x : 'L2Norm' in x),
    ('sigmoid', lambda x : 'sigmoid' in x.lower()),
    ('add-sub', lambda x : any(substr in x for substr in ['onnxruntime::cuda::OP_Add', 'onnxruntime::cuda::OP_Sub', 'at::native::AddFunctor'])),
    ('mul-div', lambda x : any(substr in x for substr in ['onnxruntime::cuda::OP_Mul', 'onnxruntime::cuda::OP_Div', 'at::native::MulFunctor', 'at::native::MulScalarFunctor', 'ScaleFunctor', 'DivFunctor', 'onnxruntime::cuda::_Scale', 'DivGrad'])),
    ('addc-muldiv', lambda x : any(substr in x for substr in ['addcdiv_cuda_kernel', 'addcmul_cuda_kernel'])),
    ('sort', lambda x : any(substr in x for substr in ['merge_sort', 'RadixSort'])),
    ('copy', lambda x : any(substr in x for substr in ['BatchedCopy', 'copy_device_to_device'])),
    ('cast', lambda x : 'OP_Cast' in x),
    ('transpose', lambda x : any(substr in x.lower() for substr in ['transpose', 'copy_kernel_impl'])),
    ('slice', lambda x : any(substr in x for substr in ['Slice'])),
    ('concat-split', lambda x : any(substr in x for substr in ['Concat', 'Split'])),
    ('expand', lambda x : any(substr in x for substr in ['Expand'])),
    ('tile', lambda x : any(substr in x for substr in ['onnxruntime::cuda::_TileMemcpyKernel'])),
    ('fill_functor', lambda x : any(substr in x for substr in ['onnxruntime::cuda::_Fill', 'at::native::FillFunctor'])),
    ('where', lambda x : any(substr in x for substr in ['onnxruntime::cuda::_TenaryElementWise', 'kernelPointwiseApply2<TensorMaskedFillOp', 'masked_fill_kernel'])),
    ('element-wise', lambda x : any(substr in x.lower() for substr in ['elementwise', 'pointwise'])),
    ('misc', lambda x : True),
]

def group_gpu_activity(lines):
    groups = { name : [] for name,_ in activities }
    for kernel_name, value in lines.items():
        for name, check in activities:
            if check(kernel_name):
                groups[name].append({kernel_name: value})
                break
    return groups

def gpu_absolute_time(activities):
    return sum([list(a.values())[0]["KernelTime"] for a in activities])

def gpu_kernel_calls(activities):
    return sum([list(a.values())[0]["Invocations"] for a in activities])

def print_summary(pt_groups, ort_groups):
    with open('parsed.csv', 'w') as csvfile:
        fieldnames = ['name', 'pt_calls', 'pt_time', 'ort_calls', 'ort_time', 'ort_faster', 'time_diff', 'contributed_perf%', 'potential_saving%']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()

        total_ort_time = 0
        total_pt_time = 0
        rows = []
        for name,_ in activities:
            pt_activities = pt_groups[name]
            ort_activities = ort_groups[name]

            row = {}
            row['name'] = name
            row['pt_calls'] = gpu_kernel_calls(pt_activities)
            row['pt_time'] = gpu_absolute_time(pt_activities)

            row['ort_calls'] = gpu_kernel_calls(ort_activities)
            row['ort_time'] = gpu_absolute_time(ort_activities)

            row['ort_faster'] = row['ort_time'] <= row['pt_time']
            
            total_ort_time += row['ort_time']
            total_pt_time += row['pt_time']

            rows.append(row)
        
        for row in rows:
            row['time_diff'] = row['ort_time'] - row['pt_time']
            if (row['ort_faster']):
                row['contributed_perf%'] = row['time_diff'] / total_ort_time * 100
            else:
                row['potential_saving%'] = row['time_diff'] / total_pt_time * 100

            writer.writerow(row)

def print_details(pt_groups, ort_groups, interested_groups):
    rows = {}
    for group in interested_groups:
        for kernel_info in ort_groups[group]:
            kernel_name = str(list(kernel_info.keys())[0])
            info = list(kernel_info.values())[0]

            if kernel_name not in rows:
                rows[kernel_name] = {}
            rows[kernel_name]['name'] = kernel_name
            rows[kernel_name]['ort_Invocations'] = info['Invocations']
            rows[kernel_name]['ort_KernelTime'] = info['KernelTime']
            rows[kernel_name]['ort_Percentage'] = info['Percentage']

        for kernel_info  in pt_groups[group]:
            kernel_name = str(list(kernel_info.keys())[0])
            info = list(kernel_info.values())[0]

            if kernel_name not in rows:
                rows[kernel_name] = {}
            rows[kernel_name]['name'] = kernel_name
            rows[kernel_name]['pt_Invocations'] = info['Invocations']
            rows[kernel_name]['pt_KernelTime'] = info['KernelTime']
            rows[kernel_name]['pt_Percentage'] = info['Percentage']


    with open('parsed_detail.csv', 'w') as csvfile:
        fieldnames = ['name', 'pt_Invocations', 'pt_KernelTime', 'pt_Percentage', 'ort_Invocations', 'ort_KernelTime', 'ort_Percentage']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows.values())


ort_kernels = get_detailed_lines('ORT', args.ort_file)
# print(json.dumps(ort_kernels, indent=4))

pt_kernels = get_detailed_lines('PT', args.pt_file)
# print(json.dumps(pt_kernels, indent=4))

pt_groups = group_gpu_activity(pt_kernels)
# print(json.dumps(pt_groups, indent=4))

ort_groups = group_gpu_activity(ort_kernels)
# print(json.dumps(ort_groups, indent=4))

interested_groups = ['element-wise', 'misc']
print_details(pt_groups, ort_groups, interested_groups)
print_summary(pt_groups, ort_groups)


