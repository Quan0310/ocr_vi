import torch
import torch.nn.functional as F
import argparse
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from CRNN import LineModel  # Sửa lại tên file model nếu khác
from PIL import Image
from preprocessor import *
from utils import *


# --------- Main Inference ---------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--model_path", type=str, default="checkpoints/model_best.pth")
    parser.add_argument("--vocab_path", type=str, default="data/vietnamese_chars.txt")
    parser.add_argument("--img_w", type=int, default=1200)
    parser.add_argument("--img_h", type=int, default=64)
    parser.add_argument("--i", action="store_true", help="Indicate if the image has white text on black background" )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load vocab
    _, idx_to_char = load_vocab(args.vocab_path)

    # Load model
    model = LineModel(img_w=args.img_w, img_h=args.img_h).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Preprocess image
    image = cv.imread(args.image_path)
    if not args.i:
        image = binarize_image(image)
    image = preprocess(image, args.img_w, args.img_h)
    image = image.unsqueeze(0)
    image = image.to(device)
    
    # Predict
    with torch.no_grad():
        output = model(image)  # [B, T, C]
        output = output.permute(1, 0, 2)  # [T, B, C]
        log_probs = F.log_softmax(output, dim=2)

        # Decode
        predicted_text = ctc_decode(log_probs.permute(1, 0, 2), idx_to_char)
        print("📝 Predicted Text:", predicted_text)
        
        plt.imshow(image[0][0].cpu().numpy(), cmap='gray')  # image: [1, 1, H, W] => [H, W]
        plt.title(predicted_text)
        plt.axis('off')
        plt.show()
if __name__ == "__main__":
    main()
