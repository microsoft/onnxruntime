import os
import re

def read_env_file(filepath):
    env_vars = {}
    with open(filepath, 'r') as f:
        for line in f:
            match = re.match(r'^(.*?)=(.*)$', line.strip())
            if match:
                env_vars[match.group(1)] = match.group(2)
    return env_vars

initial_env = read_env_file('initial_env.txt')
final_env = read_env_file('final_env.txt')

for key, value in final_env.items():
    if key not in initial_env or initial_env[key] != value:
        if key.upper() == 'PATH':
            new_paths = value.split(';')
            initial_paths = initial_env.get('PATH','').split(';')
            added_paths = [p for p in new_paths if p not in initial_paths and p]

            if added_paths:
                with open(os.environ['GITHUB_PATH'], 'a') as f:
                    for path in added_paths:
                        f.write(path + os.linesep)
        else:
            value = value.replace('%', '%25').replace('\r', '%0D').replace('\n', '%0A')
            print(f'::set-env name={key}::{value}')