import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
class LineModel(nn.Module):
    def __init__(self, img_h=64, img_w=1200, num_classes=211):
        super(LineModel, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  

            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 512x100x8
        )

        self.fc1 = nn.Linear(512 * 8, 64)
        self.gru1_fw = nn.GRU(64, 256, batch_first=True)
        self.gru1_bw = nn.GRU(64, 256, batch_first=True)
        self.bn1 = nn.BatchNorm1d(256)

        self.gru2_fw = nn.GRU(256, 256, batch_first=True)
        self.gru2_bw = nn.GRU(256, 256, batch_first=True)
        self.bn2 = nn.BatchNorm1d(512) 

        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: (B, 1, 64, 800)
        x = self.cnn(x)  
        x = x.permute(0, 3, 1, 2)  
        x = x.reshape(x.size(0), x.size(1), -1)  
        x = self.fc1(x)  
##############
        # GRU1
        out_fw1, _ = self.gru1_fw(x)  # forward
        out_bw1, _ = self.gru1_bw(torch.flip(x, dims=[1]))  # reverse time axis
        out_bw1 = torch.flip(out_bw1, dims=[1])  # flip back to original order

        out1 = out_fw1 + out_bw1  # Add
        # BatchNorm1d expects (B, C, T), so we permute
        out1_bn = self.bn1(out1.permute(0, 2, 1)).permute(0, 2, 1)

        # GRU2
        out_fw2, _ = self.gru2_fw(out1_bn)
        out_bw2, _ = self.gru2_bw(torch.flip(out1_bn, dims=[1]))
        out_bw2 = torch.flip(out_bw2, dims=[1])

        out2 = torch.cat([out_fw2, out_bw2], dim=-1)  # Concatenate last dim
        out2_bn = self.bn2(out2.permute(0, 2, 1)).permute(0, 2, 1)

        x = self.fc2(out2_bn) 
        y_pred = F.softmax(x, dim=2) 
        return x
if __name__ == "__main__":
    model = LineModel()
    input = torch.randn(2, 1, 64, 800)
    output = model(input)
    summary(model, input_size=(2, 1, 64, 1200))
    print(output.shape)