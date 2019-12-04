import pygit2
import argparse
import json

def format_component(submod):
    return {"component":{"type":"git","git":{"commitHash":str(submod.head_id), "repositoryUrl":str(submod.url)}}}

def process_component(repo):
    return [repo.lookup_submodule(submod) for submod in repo.listall_submodules()]

def recursive_process(base_repo):
    processed_subs = []
    repos_to_process = [base_repo]
    while repos_to_process:
        repo = repos_to_process.pop()
        submodules = process_component(repo)
        processed_subs.extend(submodules)
        repos_to_process.extend([mod.open() for mod in submodules])
    return {"Registrations":[format_component(component) for component in processed_subs]}

def main(repo_path, output_file):
    repo = pygit2.Repository(repo_path)
    registrations = recursive_process(repo)
    with open(output_file, 'w') as f:
        json.dump(registrations, f, indent=4, sort_keys=True)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("base_repository", help="path to base repository to get registrations for.")
    parser.add_argument("-o", "--output", help="output file name.", default="cgmanifest.json")
    args = parser.parse_args()
    main(args.base_repository, args.output)

