import argparse
import os
from typing import Optional

import cv2
import open_clip
import torch
from PIL import Image
from sentence_transformers import util


def arg_parser():
    parser = argparse.ArgumentParser(description="Options for Compare 2 image")
    parser.add_argument("--image1", type=str, help="Path to image 1")
    parser.add_argument("--image2", type=str, help="Path to image 2")
    parser.add_argument("--cache_dir", type=str, help="Path to model cache directory")
    args = parser.parse_args()
    return args


def image_encoder(img: Image.Image, cache_dir: Optional[str] = None):  # -> torch.Tensor:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16-plus-240", pretrained="laion400m_e32", cache_dir=cache_dir
    )
    model.to(device)

    img1 = Image.fromarray(img).convert("RGB")
    img1 = preprocess(img1).unsqueeze(0).to(device)
    img1 = model.encode_image(img1)
    return img1


def load_image(image_path: str):  # -> Image.Image:
    # cv2.imread() can silently fail when the path is too long
    # https://stackoverflow.com/questions/68716321/how-to-use-absolute-path-in-cv2-imread
    if os.path.isabs(image_path):
        directory = os.path.dirname(image_path)
        current_directory = os.getcwd()
        os.chdir(directory)
        img = cv2.imread(os.path.basename(image_path), cv2.IMREAD_UNCHANGED)
        os.chdir(current_directory)
    else:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    return img


def generate_score(image1: str, image2: str, cache_dir: Optional[str] = None):  # -> float:
    test_img = load_image(image1)
    data_img = load_image(image2)
    img1 = image_encoder(test_img, cache_dir)
    img2 = image_encoder(data_img, cache_dir)
    cos_scores = util.pytorch_cos_sim(img1, img2)
    score = round(float(cos_scores[0][0]) * 100, 2)
    return score


def main():
    args = arg_parser()
    image1 = args.image1
    image2 = args.image2
    cache_dir = args.cache_dir
    score = round(generate_score(image1, image2, cache_dir), 2)
    print("similarity Score: ", {score})
    if score < 97:
        print(f"{image1} and {image2} are different")
        raise SystemExit(1)
    else:
        print(f"{image1} and {image2} are same")


if __name__ == "__main__":
    main()
