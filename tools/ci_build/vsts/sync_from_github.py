#!/usr/bin/env python3

import argparse
import contextlib
import logging
import os
import shutil
import subprocess

logging.basicConfig(format="%(asctime)s %(name)s [%(levelname)s] - %(message)s", level=logging.DEBUG)
log = logging.getLogger()

REPO_URL = "https://aiinfra.visualstudio.com/Lotus/_git/onnxruntime"
GITHUB_REPO_URL = "https://github.com/Microsoft/onnxruntime.git"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to automate syncing code from GitHub repo. "
                    "Clones the repo, merges the GitHub source branch into the target branch, "
                    "and then pushes target branch.")

    parser.add_argument("--github-source-branch-name", required=True,
                        help="The name of the upstream GitHub branch to sync from.")
    parser.add_argument("--target-branch-name", required=True,
                        help="The name of the target branch to which the GitHub changes are merged.")

    parser.add_argument("--token-env",
                        help="The environment variable containing the access token value.")

    parser.add_argument("--work-dir", default=".",
                        help="The path to the working directory (e.g., where to put the cloned repo).")
    parser.add_argument("--retain-work-repo", action="store_true",
                        help="Set this to keep the working repo around (e.g., for debugging).")

    parser.add_argument("--git-user-name",
                        help="The Git username (user.name config) to use in the working repo.")
    parser.add_argument("--git-user-email",
                        help="The Git user email (user.email config) to use in the working repo.")

    args = parser.parse_args()

    # get token value
    if args.token_env is None or args.token_env not in os.environ:
        log.warning("No access token value available.")
        args.token = None
    else:
        args.token = os.environ[args.token_env]

    # get absolute work dir path
    args.work_dir = os.path.realpath(args.work_dir)

    return args

def git(*git_args, check=True, capture=False, quiet=False):
    """Runs a git command.

    Args:
        *git_args - arguments to git
        check - whether to fail on error
        capture - whether to capture stdout and stderr
        quiet - whether to log arguments and display output if not capturing

    Returns:
        A subprocess.CompletedProcess instance.
    """
    args = ["git"] + list(git_args)
    if not quiet:
        log.debug("Running command: %s", args)
        if capture:
            log.debug("Capturing output.")
    output = subprocess.PIPE if capture else (subprocess.DEVNULL if quiet else None)
    return subprocess.run(args, check=check, stdout=output, stderr=output)

@contextlib.contextmanager
def scoped_cwd(new_dir):
    """Change to new_dir on entry, change back to original directory on exit."""
    old_dir = os.getcwd()
    try:
        os.chdir(new_dir)
        yield
    finally:
        os.chdir(old_dir)

def check_for_changes(target_branch_name):
    """Returns whether there are changes that require a push to the remote repo."""
    # diff with remote target branch
    target_diff_result = git("diff", "--quiet", "--exit-code",
                             "refs/remotes/origin/{}".format(target_branch_name), "refs/heads/{}".format(target_branch_name),
                             check=False)
    if target_diff_result.returncode == 0:
        logging.debug("Remote target branch is identical to target branch.")
        return False

    return True

def merge_from_github(work_dir, target_branch_name, github_source_branch_name, retain_repo,
                      git_user_name, git_user_email, git_auth_token):
    """Performs the merge in a locally cloned repo.

    Returns true if there are changes that have been pushed, false otherwise.
    """
    with contextlib.ExitStack() as context_stack:
        if git_auth_token is not None:
            # set auth token in Git config and clean up later
            config_name = "http.{}.extraheader".format(REPO_URL)
            logging.debug("Setting auth token in Git config ({}).".format(config_name))
            git("config", "--global", config_name, "Authorization: Bearer {}".format(git_auth_token),
                quiet=True)
            context_stack.callback(git, "config", "--global", "--unset-all", config_name)

        os.makedirs(work_dir, exist_ok=True)
        local_repo_dir = os.path.join(work_dir, "local_sync_repo")
        git("clone", "--branch", target_branch_name, "--", REPO_URL, local_repo_dir)

        if not retain_repo:
            # adapted from here: https://docs.python.org/3.6/library/shutil.html?highlight=shutil#rmtree-example
            def rmtree_remove_readonly(function, path, excinfo):
                import stat
                os.chmod(path, stat.S_IWRITE)
                function(path)

            context_stack.callback(shutil.rmtree, local_repo_dir, onerror=rmtree_remove_readonly)

        # cd local_repo_dir
        context_stack.enter_context(scoped_cwd(local_repo_dir))

        if git_user_name is not None:
            git("config", "user.name", git_user_name)
        if git_user_email is not None:
            git("config", "user.email", git_user_email)

        git("remote", "add", "github", GITHUB_REPO_URL)
        git("fetch", "github", "refs/heads/{}".format(github_source_branch_name))

        git("checkout", "-B", target_branch_name, "refs/heads/{}".format(target_branch_name))

        git("merge", "refs/remotes/github/{}".format(github_source_branch_name))

        changes_detected = check_for_changes(target_branch_name)

        # only push if there are changes
        if changes_detected:
            git("push", "origin", "refs/heads/{}".format(target_branch_name))

        return changes_detected

def main():
    args = parse_args()

    log.info("Syncing code from GitHub - GitHub source branch: %s, target branch: %s",
             args.github_source_branch_name, args.target_branch_name)

    changes_pushed = merge_from_github(work_dir=args.work_dir,
                                       target_branch_name=args.target_branch_name,
                                       github_source_branch_name=args.github_source_branch_name,
                                       retain_repo=args.retain_work_repo,
                                       git_user_name=args.git_user_name, git_user_email=args.git_user_email,
                                       git_auth_token=args.token)

    if not changes_pushed:
        log.info("No changes were pushed.")

    log.info("Sync complete.")

if __name__ == "__main__":
    import sys
    sys.exit(main())
