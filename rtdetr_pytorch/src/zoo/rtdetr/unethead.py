import torch.nn as nn
import torch
from src.core import register

@register
class UNetHead(nn.Module):
    def __init__(self, num_classes):
        super(UNetHead, self).__init__()
        # self.upscale = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # self.upscale = torch.nn.functional.interpolate(x2, scale_factor=2, mode='bilinear')
        self.upconv = nn.ConvTranspose2d(256*3, 128, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.upconv3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x_enc):
        # x1 = self.upconv1(x_enc[2])  # 20x20

        x1 = torch.nn.functional.interpolate(x_enc[1], scale_factor=2, mode='bilinear')
        x2 = torch.nn.functional.interpolate(x_enc[2], scale_factor=4, mode='bilinear')
        x1 = torch.cat([x_enc[0],x1,x2], dim=1)  # Concatenate with 40x40
        x1 = self.upconv(x1)
        x1 = self.conv1(x1)
        x1 = self.relu(x1)
        x2 = self.upconv2(x1)  # 40x40
        # x2 = torch.cat((x2, x_enc[0]), dim=1)  # Concatenate with 80x80
        x2 = self.conv2(x2)
        x2 = self.relu(x2)
        x2 = self.upconv3(x2)  # Final output
        x2 = self.conv2(x2)
        x2 = self.relu(x2)
        out = self.final_conv(x2)  # Final output
        # import pdb; pdb.set_trace()    

        return out
