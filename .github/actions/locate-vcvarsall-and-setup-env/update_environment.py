import os
import re


def read_env_file(filepath):
    env_vars = {}
    with open(filepath) as f:
        for line in f:
            match = re.match(r"^(.*?)=(.*)$", line.strip())
            if match:
                env_vars[match.group(1).upper()] = match.group(2)
    return env_vars


initial_env = read_env_file("initial_env.txt")
final_env = read_env_file("final_env.txt")

for key, value in final_env.items():
    if key not in initial_env or initial_env[key] != value:
        if key.startswith("_"):
            continue
        if key.upper() == "PATH":
            new_paths = value.split(";")
            initial_paths = initial_env.get("PATH", "").split(";")
            added_paths = [p for p in new_paths if p not in initial_paths and p]

            if added_paths:
                print("Adding paths")
                with open(os.environ["GITHUB_PATH"], "a") as f:
                    for path in added_paths:
                        print(f"Adding PATH: {path}")
                        f.write(path + os.linesep)
        else:
            # Use GITHUB_ENV
            with open(os.environ["GITHUB_ENV"], "a") as f:
                print(f"Setting {key}={value}\n")
                f.write(f"{key}={value}\n")
