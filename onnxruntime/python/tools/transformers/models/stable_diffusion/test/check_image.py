import argparse

import cv2
import open_clip
import torch
from PIL import Image
from sentence_transformers import util


def arg_parser():
    parser = argparse.ArgumentParser(description="Options for Compare 2 image")
    parser.add_argument("--image1", type=str, help="Path to image 1")
    parser.add_argument("--image2", type=str, help="Path to image 2")
    args = parser.parse_args()
    return args

def imageEncoder(img):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
    model.to(device)

    img1 = Image.fromarray(img).convert('RGB')
    img1 = preprocess(img1).unsqueeze(0).to(device)
    img1 = model.encode_image(img1)
    return img1

def generateScore(image1, image2):
    test_img = cv2.imread(image1, cv2.IMREAD_UNCHANGED)
    data_img = cv2.imread(image2, cv2.IMREAD_UNCHANGED)
    img1 = imageEncoder(test_img)
    img2 = imageEncoder(data_img)
    cos_scores = util.pytorch_cos_sim(img1, img2)
    score = round(float(cos_scores[0][0])*100, 2)
    return score

def main():
    args = arg_parser()
    image1 = args.image1
    image2 = args.image2

    score = round(generateScore(image1, image2), 2)
    if score < 99:
        print(f"score is{score}, Images are different")
        raise SystemExit(1)

if __name__ == "_main__":
    main()
