import torch
import cv2
def custom_collate_fn(batch):
    # batch = list of (img, label, label_len, text)
    imgs, labels, label_lens, texts = zip(*batch)

    # Stack ảnh vào batch
    imgs = torch.stack(imgs, dim=0)  # [B, 1, H, W]

    # Nối các label lại thành 1 tensor dài (cho CTC loss)
    labels = torch.cat(labels, dim=0)  # [total_label_len]

    # Gộp độ dài từng label
    label_lens = torch.tensor(label_lens, dtype=torch.long)  # [B]

    return imgs, labels, label_lens, texts

# --------- Load vocabulary ----------
def load_vocab(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        alphabet = f.readline().rstrip("\n")
    char_to_idx = {char: idx + 1 for idx, char in enumerate(alphabet)}  # 0 = blank
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    return char_to_idx, idx_to_char

# --------- Decode prediction using CTC rules ----------
def ctc_decode(preds, idx_to_char):
    preds = preds.argmax(dim=2)  # (B, T)
    preds = preds[0].detach().cpu().numpy()  # batch = 1
    decoded = []
    prev = -1
    for p in preds:
        if p != prev and p != 0:  # 0 = blank
            decoded.append(p)
        prev = p
    return ''.join([idx_to_char[i] for i in decoded])

def binarize_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    alpha = 1.5  # Độ tương phản (1.0-3.0)
    beta = 50    # Độ sáng (0-100)
    adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    _, binary = cv2.threshold(adjusted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    denoised = cv2.GaussianBlur(binary, (3, 3), 0)
    color_denoised = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    return color_denoised