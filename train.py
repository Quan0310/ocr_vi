import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import *
from CRNN import LineModel
from dataset import HandwritingDataset
from tqdm import tqdm  # để hiển thị tiến độ đẹp
from tensorboardX import SummaryWriter
def train():
    parser = argparse.ArgumentParser("Training model")
    parser.add_argument("--img_w", type=int, default=1200)
    parser.add_argument("--img_h", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--saved_path", type=str, default="checkpoints")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.cuda.manual_seed(1989)
    else:
        torch.manual_seed(1989)
    training_params = {
        "batch_size": args.batch_size,
        "num_workers": 4,
        "shuffle": True,
        "drop_last": True,
        "collate_fn": custom_collate_fn
    }
    test_params = {
        "batch_size": args.batch_size,
        "num_workers": 4,
        "shuffle": False,
        "drop_last": False,
        "collate_fn": custom_collate_fn
    }

    # Dataset
    train_dataset = HandwritingDataset(
        root_dir=args.data_path,
        mode="train",
        img_w=args.img_w,
        img_h=args.img_h,
        max_text_len=74
    )
    val_dataset = HandwritingDataset(
        root_dir=args.data_path,
        mode="val",
        img_w=args.img_w,
        img_h=args.img_h,
        max_text_len=74
    )

    # Dataloader
    train_loader = DataLoader(train_dataset, **training_params)
    val_loader = DataLoader(val_dataset, **test_params)

    # Model, Loss, Optimizer
    model = LineModel(img_w=args.img_w, img_h=args.img_h).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    os.makedirs(args.saved_path, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.saved_path, "logs"))

    start_epoch = 0
    best_val_loss = float("inf")

    checkpoint_path = os.path.join(args.saved_path, "model_latest.pth")
    if args.resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
        print(f"🔄 Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}", leave=False)

        for images, labels, labels_lengths, _ in loop:
            images = images.to(device)
            labels = labels.to(device)
            labels_lengths = labels_lengths.to(device)

            outputs = model(images)  # (B, T, C)
            outputs = outputs.permute(1, 0, 2)  # (T, B, C)
            outputs = F.log_softmax(outputs, dim=2) 

            input_lengths = torch.full((images.size(0),), fill_value=outputs.size(0), dtype=torch.long).to(device)

            loss = criterion(outputs, labels, input_lengths, labels_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] Training Loss: {avg_train_loss:.4f}")
        writer.add_scalar("Loss/Train", avg_train_loss, epoch + 1)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels, labels_lengths, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                labels_lengths = labels_lengths.to(device)

                outputs = model(images)
                outputs = outputs.permute(1, 0, 2)
                outputs = F.log_softmax(outputs, dim=2) 
                input_lengths = torch.full((images.size(0),), fill_value=outputs.size(0), dtype=torch.long).to(device)

                loss = criterion(outputs, labels, input_lengths, labels_lengths)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"[Epoch {epoch+1}] Validation Loss: {avg_val_loss:.4f}")
        writer.add_scalar("Loss/Val", avg_val_loss, epoch + 1)

        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss
        }, os.path.join(args.saved_path, "model_latest.pth"))

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(args.saved_path, "model_best.pth"))
            print(f"✅ Saved best model at epoch {epoch+1} with val loss: {best_val_loss:.4f}")
        
    writer.close()

if __name__ == "__main__":
    train()
