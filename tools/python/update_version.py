import os


def update_version():
    version = ""
    cwd = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(cwd, "..", "..", "VERSION_NUMBER")) as f:
        version = f.readline().strip()
    lines = []
    current_version = ""
    file_path = os.path.join(cwd, "..", "..", "docs", "Versioning.md")
    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("|"):
                sections = line.split("|")
                if len(sections) == 8 and sections[1].strip()[0].isdigit():
                    current_version = sections[1].strip()
                    break
    print("Current version of ORT seems to be: " + current_version)
    if version != current_version:
        with open(file_path, "w") as f:
            for i, line in enumerate(lines):
                f.write(line)
                if line.startswith("|--"):
                    sections = lines[i + 1].split("|")
                    # Make sure there are no 'False Positive' version additions
                    # by making sure the line we are building a new line from
                    # contains the current_version
                    if len(sections) > 1 and sections[1].strip() == current_version:
                        sections[1] = " " + version + " "
                        new_line = "|".join(sections)
                        f.write(new_line)
    lines = []
    current_version = ""
    file_path = os.path.join(cwd, "..", "..", "docs", "python", "README.rst")
    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            sections = line.strip().split(".")
            if len(sections) == 3 and sections[0].isdigit() and sections[1].isdigit() and sections[2].isdigit():
                current_version = line.strip()
                break
    if version != current_version:
        inserted = False
        with open(file_path, "w") as f:
            for line in lines:
                sections = line.strip().split(".")
                if (
                    inserted is False
                    and len(sections) == 3
                    and sections[0].isdigit()
                    and sections[1].isdigit()
                    and sections[2].isdigit()
                ):
                    f.write(version + "\n")
                    f.write("^" * len(version) + "\n\n")
                    f.write(
                        "Release Notes : https://github.com/Microsoft/onnxruntime/releases/tag/v"
                        + version.strip()
                        + "\n\n"
                    )
                    inserted = True
                f.write(line)
    lines = []
    current_version = ""
    file_path = os.path.join(cwd, "..", "..", "onnxruntime", "__init__.py")
    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("__version__"):
                current_version = line.split("=")[1].strip()[1:-1]
                break
    if version != current_version:
        with open(file_path, "w") as f:
            for line in lines:
                if line.startswith("__version__"):
                    f.write('__version__ = "' + version + '"\n')
                    continue
                f.write(line)

    # update version for NPM packages
    current_version = ""
    js_root = os.path.join(cwd, "..", "..", "js")

    def run(args, cwd):
        from util import is_windows, run

        if is_windows():
            args = ["cmd", "/c", *args]
        run(*args, cwd=cwd)

    # check if node, npm and yarn are installed
    run(["node", "--version"], cwd=js_root)
    run(["npm", "--version"], cwd=js_root)
    run(["yarn", "--version"], cwd=js_root)

    # upgrade version for onnxruntime-common
    run(["npm", "version", version], cwd=os.path.join(js_root, "common"))
    run(["npm", "install", "--package-lock-only", "--ignore-scripts"], cwd=os.path.join(js_root, "common"))

    # upgrade version for onnxruntime-node
    run(["npm", "version", version], cwd=os.path.join(js_root, "node"))
    run(["npm", "install", "--package-lock-only", "--ignore-scripts"], cwd=os.path.join(js_root, "node"))

    # upgrade version for onnxruntime-web
    run(["npm", "version", version], cwd=os.path.join(js_root, "web"))
    run(["npm", "install", "--package-lock-only", "--ignore-scripts"], cwd=os.path.join(js_root, "web"))

    # upgrade version for onnxruntime-react-native
    run(["npm", "version", version], cwd=os.path.join(js_root, "react_native"))
    run(["yarn", "upgrade", "onnxruntime-common"], cwd=os.path.join(js_root, "react_native"))

    # upgrade version.ts in each package
    run(["npm", "ci"], cwd=js_root)
    run(["npm", "run", "update-version", "common"], cwd=js_root)
    run(["npm", "run", "update-version", "node"], cwd=js_root)
    run(["npm", "run", "update-version", "web"], cwd=js_root)
    run(["npm", "run", "update-version", "react_native"], cwd=js_root)
    run(["npm", "run", "format"], cwd=js_root)


if __name__ == "__main__":
    update_version()
