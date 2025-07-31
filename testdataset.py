from torch.utils.data import DataLoader
from dataset import HandwritingDataset
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from utils import *


if __name__ == "__main__":
    training_params = {
        "batch_size": 8,
        "num_workers": 4,  # có thể đặt về 0 để debug trước
        "shuffle": True,
        "drop_last": False,
        "collate_fn": custom_collate_fn
    }

    train_dataset = HandwritingDataset(
        root_dir="data", 
        mode="train",
        img_w=1200,
        img_h=64,
        max_text_len=75
    )

    training_loader = DataLoader(train_dataset, **training_params)

    for batch in training_loader:
        imgs, labels, label_lens, texts = batch
        print(imgs.shape)
        print(labels)
        print(label_lens)
        print(texts[0])
        plt.imshow(imgs[0].squeeze().numpy(), cmap='gray')
        plt.title(texts[0])  # nếu bạn muốn xem text đi kèm
        plt.show()
        break  # test 1 batch