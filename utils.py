import torch
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