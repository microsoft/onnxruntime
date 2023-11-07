import os
import subprocess
import sys



def is_windows():
    return sys.platform.startswith("win")


def run_subprocess(args, cwd=None, capture=False, dll_path=None, shell=False, env={}, log=None):  # noqa: B006
    if log:
        log.info(f"Running subprocess in '{cwd or os.getcwd()}'\n{args}")
    my_env = os.environ.copy()
    if dll_path:
        if is_windows():
            my_env["PATH"] = dll_path + os.pathsep + my_env["PATH"]
        else:
            if "LD_LIBRARY_PATH" in my_env:
                my_env["LD_LIBRARY_PATH"] += os.pathsep + dll_path
            else:
                my_env["LD_LIBRARY_PATH"] = dll_path

    stdout, stderr = (subprocess.PIPE, subprocess.STDOUT) if capture else (None, None)
    my_env.update(env)
    completed_process = subprocess.run(args, cwd=cwd, check=True, stdout=stdout, stderr=stderr, env=my_env, shell=shell)

    if log:
        log.debug("Subprocess completed. Return code=" + str(completed_process.returncode))
    return completed_process
