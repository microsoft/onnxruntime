#!/usr/bin/python

import argparse
import json
import pandas as pd
import subprocess as sp


def _demangle(name, demangler='cu++filt'):
    try:
        p = sp.Popen([demangler, name], stdin=sp.PIPE, stdout=sp.PIPE)
        out, _ = p.communicate()
        return out.decode('utf-8').strip()
    except:
        return name


def _get_args():
    parser = argparse.ArgumentParser(description='onnxruntime bench tool')
    parser.add_argument('input', type=str, help='Trace input file, formatted as JSON')
    parser.add_argument('--demangler', required=False, type=str, default='cu++filt', help='The command to use to demangle C++ identifiers')
    parser.add_argument('--shape-sensitive', action='store_true',
                        help='Perform a shape sensitive analysis of kernel execution times')

    parser.add_argument('--dimension-sensitive', action='store_true',
                        help='Perform a kernel launch dimension sensitive analysis of kernel execution times')

    parser.add_argument('--filter', type=str, nargs='+', action='extend', help='Restrict analysis to the specified identifiers, i.e., specify a filter list')
    parser.add_argument('--csv', help='save data to csv')
    parser.add_argument('-c', '--count', type=int, default=40, help='list top N items')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose')
    args = parser.parse_args()
    return args

def _shape_to_string(shape):
    res = ''
    for d in shape:
        if len(d) > 1:
            raise ValueError('Unhandled type in _shape_to_string()')
        key = list(d.keys())[0]
        value = list(d.values())[0]
        if len(res) != 0:
            res += '__'
        res += f'{key}_{"x".join(str(v) for v in value)}'
    return res


def _json_to_df(profile_path, filter_set):
    cpu_entries = []
    gpu_entries = []
    
    with open(profile_path, "r") as f:
        data = json.load(f)
    if type(data) == dict:
        data = data['traceEvents']

    for idx, item in enumerate(data):
        cat = item.get('cat')
        if cat is None:
            continue
        dur = item.get('dur')
        if dur is None:
            continue
        arg = item.get('args')
        if arg is None:
            continue
        op = arg.get("op_name")

        name = item['name']

        if filter_set is not None and len(filter_set) > 0 and name not in filter_set and op not in filter_set:
            continue
        
        if cat != 'Kernel' and not name.endswith('kernel_time'):
            continue

        if cat == 'Kernel':
            gpu_entries.append({
                'name': name,
                'duration': dur,
                'dimensions': f'{item["args"]["block_x"]}_{item["args"]["block_y"]}_{item["args"]["block_z"]}_{item["args"]["grid_x"]}_{item["args"]["grid_y"]}_{item["args"]["grid_z"]}',
                'op_name': op,
            })
        else:
            cpu_entries.append({
                'name': item['args']['op_name'],
                'duration': dur,
                'input_shape': _shape_to_string(item['args']['input_type_shape']),
                'output_shape': _shape_to_string(item['args']['output_type_shape']),
            })
                
    cpu_df = pd.DataFrame(cpu_entries)
    gpu_df = pd.DataFrame(gpu_entries)
    cpu_df['count'] = 1
    gpu_df['count'] = 1
    return cpu_df, gpu_df

def _print_cpu_top_hitters(df, args):
    top = args.count
    group_key = ['name', 'input_shape'] if args.shape_sensitive else ['name']
    df2 = df[['duration', 'count']].sum()
    df['pct'] = 100 * (df['duration'] / df2['duration'])
    fields = group_key + ['duration', 'pct', 'count']
    df1 = df[fields].groupby(group_key).sum().reset_index()
    df1 = df1.sort_values(by='duration', ascending=False)[:top]
    df1['cumulative_pct'] = df1['pct'].cumsum()
    df1['cumulative_dur'] = df1['duration'].cumsum()
    print('\n------ Top CPU Kernel Times ------')
    print(df1.round(2).to_string(index=False))
    if args.csv:
        df1.to_csv(f'{args.csv}_cpu_kernel_times.csv', index=False)

def _print_gpu_top_hitters(df, args):
    top = args.count
    group_key = ['name', 'dimensions'] if args.dimension_sensitive else ['name']
    df2 = df[['duration', 'count']].sum()
    df['pct'] = 100 * (df['duration'] / df2['duration'])
    fields = group_key + ['duration', 'pct', 'count']
    df1 = df[fields].groupby(group_key).sum().reset_index()
    df1 = df1.sort_values(by='duration', ascending=False)[:top]
    df1['cumulative_pct'] = df1['pct'].cumsum()
    df1['cumulative_dur'] = df1['duration'].cumsum()
    df1['name'] = df1['name'].apply(lambda x: _demangle(x, args.demangler))
    print('\n------ Top GPU Kernel Times ------')
    print(df1.round(2).to_string(index=False))
    if args.csv:
        df1.to_csv(f'{args.csv}_gpu_kernel_times.csv', index=False)

def main():
    args = _get_args()
    filter_set = set(args.filter if args.filter is not None else [])

    cpu_df, gpu_df = _json_to_df(args.input, filter_set)

    pd.set_option('display.max_colwidth', 120)
    _print_cpu_top_hitters(cpu_df, args)
    _print_gpu_top_hitters(gpu_df, args)


if __name__ == '__main__':
    main()
