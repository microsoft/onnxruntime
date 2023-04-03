import json
import os
import tarfile

import wget


def get_tar_file(link):
    file_name = link.split("/")[-1]
    return file_name


def create_model_folder(model):
    os.mkdir(model)


def extract_and_get_files(file_name):
    model_folder = file_name.replace(".tar.gz", "") + "/"
    create_model_folder(model_folder)
    model_tar = tarfile.open(file_name)
    model_tar.extractall(model_folder)
    file_list = model_tar.getnames()
    file_list.sort()
    model_tar.close()
    return model_folder, file_list


def download_model(link):
    file_name = get_tar_file(link)
    wget.download(link)
    model_folder, file_list = extract_and_get_files(file_name)
    return model_folder, file_list


def get_model_path(file_list):
    for file_name in file_list:
        if ".onnx" in file_name:
            return file_name


def get_test_path(model_path):
    model_filename = os.path.basename(model_path)
    test_path = model_path.split(model_filename)[0]
    return test_path


def create_model_object(model, folder, model_file_path, test_path):
    model_dict = {}
    model_dict["model_name"] = model
    model_dict["working_directory"] = "./models/" + folder
    model_dict["model_path"] = "./" + model_file_path
    model_dict["test_data_path"] = "./" + test_path
    return model_dict


def get_model_info(link):
    model_folder, file_list = download_model(link)
    model = model_folder[:-1]
    model_file_path = get_model_path(file_list)
    test_path = get_test_path(model_file_path)
    model_info = create_model_object(model, model_folder, model_file_path, test_path)
    return model_info


def write_json(models):
    model_json = json.dumps(models, indent=4)
    with open("model_list.json", "w") as fp:
        fp.write(model_json)


def main():
    links = []
    with open("links.txt") as fh:
        links = [link.rstrip() for link in fh.readlines()]

    model_list = []
    for link in links:
        model_list.append(get_model_info(link))
    write_json(model_list)


if __name__ == "__main__":
    main()
