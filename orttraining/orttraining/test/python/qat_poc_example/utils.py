import logging
import os


def makedir(dir_path):
    """Creates a directory if one does not already exist"""
    if not os.path.exists(dir_path):
        logging.info(f"Creating directory: {dir_path}")
        os.makedirs(dir_path, exist_ok=True)
    else:
        logging.info(f"Directory already exists: {dir_path}. Will overwrite existing *.onnx and *.ckpt files.")
