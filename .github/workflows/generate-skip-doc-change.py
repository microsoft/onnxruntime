# generate_skip_doc_change.py
import os
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, Template

GITHUB_DIR = Path(__file__).resolve().parent.parent


class Skipped_Workflow:
    def __init__(self, workflow_name: str, job_names: list, output_file_name: str):
        self.workflow_name = workflow_name
        self.job_names = job_names
        self.fake_file_name = output_file_name


WIN_GPU_CI_WORKFLOW = Skipped_Workflow(
    workflow_name="Windows GPU CI Pipeline",
    job_names=[
        "cuda build_x64_RelWithDebInfo",
        "dml build_x64_RelWithDebInfo",
        "training build_x64_RelWithDebInfo",
        "kernelDocumentation build_x64_RelWithDebInfo",
    ],
    output_file_name=str(GITHUB_DIR.joinpath("workflows/generated_fake_win_gpu_ci.yml")),
)


def generate_fake_ci_yaml(template: Template, workflow: Skipped_Workflow):
    py_file_name = os.path.basename(__file__)
    content = template.render(
        python_file_name=py_file_name, ci_workflow_name=workflow.workflow_name, job_names=workflow.job_names
    )

    filename = workflow.fake_file_name
    with open(filename, mode="w", encoding="utf-8") as output_file:
        output_file.write(content)
        if content[-1] != "\n":
            output_file.write("\n")
        print(f"... wrote {filename}")


def main() -> None:
    environment = Environment(loader=FileSystemLoader(str(GITHUB_DIR.joinpath("workflows/"))))
    template = environment.get_template("skip-doc-change.yml.j2")
    skipped_workflows = [WIN_GPU_CI_WORKFLOW]
    [generate_fake_ci_yaml(template, workflow) for workflow in skipped_workflows]


if __name__ == "__main__":
    main()
