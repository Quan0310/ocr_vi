import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from data_augment import *
from preprocessor import *

class HandwritingDataset(Dataset):
    def __init__(self, root_dir, mode, img_w, img_h, max_text_len):
        assert mode in ["train", "val"], "mode must be either 'train' or 'val'"
        
        self.data_dir = os.path.join(root_dir, "lines")
        id_file = os.path.join(root_dir, "splits", f"{mode}.txt")
        
        with open(id_file, 'r') as f:
            self.ids = [line.strip() for line in f.readlines()]
        
        vocab_path = os.path.join(root_dir, "vietnamese_chars.txt")
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.ALPHABET = f.readline().rstrip("\n")

        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(self.ALPHABET)}  # Start from 1 (0 = CTC blank)
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}

        self.img_w = img_w
        self.img_h = img_h
        self.max_text_len = max_text_len

    def __len__(self):
        return len(self.ids)

    def text_to_labels(self, text):
        return [self.char_to_idx[char] for char in text if char in self.char_to_idx]

    def labels_to_text(self, labels):
        return ''.join([self.idx_to_char[label] for label in labels if label in self.idx_to_char])
    
    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        img_path = os.path.join(self.data_dir, sample_id + ".png")
        label_path = os.path.join(self.data_dir, sample_id + ".txt")

        img = cv.imread(img_path)

        # Augmentation chỉ áp dụng khi train
        if hasattr(self, 'mode') and self.mode == "train":
            img = augment_image(img)

        # Padding, chuyển grayscale, normalize
        img = preprocess(img, self.img_w, self.img_h)

        with open(label_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        label = torch.LongTensor(self.text_to_labels(text[:self.max_text_len]))
        label_length = torch.tensor(len(label), dtype=torch.long)
        return img, label, label_length, text