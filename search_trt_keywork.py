import os

# 递归遍历目录下的所有文件
def list_files(dir_path):
    file_list = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".yml") or file.endswith(".yaml"):
                file_list.append(os.path.join(root, file))
    return file_list

# 搜索文件中的特定关键词
def search_keywords(file_path, keywords):
    result = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if any(keyword.lower() in line.lower() for keyword in keywords):
                result.append((i + 1, line.strip()))
    return result

# 主函数
def main():
    dir_path = "tools/ci_build/github/azure-pipelines/templates"
    output_file = "output.md"
    keywords = ["trt", "tensorrt"]
    file_list = list_files(dir_path)
    
    repo_url = "https://github.com/microsoft/onnxruntime/blob/master/"

    with open(output_file, "w", encoding="utf-8") as md_file:
        for file_path in file_list:
            search_result = search_keywords(file_path, keywords)
            if search_result:
                md_file.write(f"## {os.path.relpath(file_path, dir_path)}\n")
                for line_num, line in search_result:
                    relative_file_path = os.path.relpath(file_path, dir_path)
                    file_url = f"{repo_url}{relative_file_path}#L{line_num}"
                    md_file.write(f"- Line {line_num}: [`{line}`]({file_url})\n")
                md_file.write("\n")

if __name__ == "__main__":
    main()
