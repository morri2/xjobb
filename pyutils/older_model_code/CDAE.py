# A Convolutional Denoising Auto-Encoder a la *D. Lee et al*
# WIP - very sus 
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torchinfo


class CDAE(nn.Module):
    def __init__(self, bottle_neck_channels=16):
        super(CDAE, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.encode1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=bottle_neck_channels * 2 ** 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
          )
        
        self.encode2 = nn.Sequential(
            nn.Conv2d(in_channels=bottle_neck_channels * 2 ** 2, out_channels=bottle_neck_channels * 2 ** 1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
          )
        
        self.encode3 = nn.Sequential(
            nn.Conv2d(in_channels=bottle_neck_channels * 2 ** 1, out_channels=bottle_neck_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
            
          )
        
        self.decode1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=bottle_neck_channels, out_channels=bottle_neck_channels * 2 ** 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
            
        )

        self.decode2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2 * (bottle_neck_channels * 2 ** 1), out_channels=bottle_neck_channels * 2 ** 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
            
        )

        self.decode3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2 * (bottle_neck_channels * 2 ** 2), out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
            
        )



    def forward(self, x):
        encoding1 = self.encode1(x)
        encoding2 = self.encode2(encoding1)
        encoding3 = self.encode3(encoding2)
        decoding1 = self.decode1(encoding3)
        decoding1_with_skip = torch.cat((decoding1, encoding2), 1)
        decoding2 = self.decode2(decoding1_with_skip)
        decoding2_with_skip = torch.cat((decoding2, encoding1), 1)
        out = self.decode3(decoding2_with_skip)
        return out

    
# Example usage
if __name__ == "__main__":
    model = CDAE()
    x = torch.randn((1, 1, 320, 320))  # Example input tensor (Batch, Width, Height)
    t = time.time()
    out = model(x)

    torchinfo.summary(model, x.shape)

    print("time =", time.time() - t)
    print("in shape", x.shape)  # Should be (1, 256, 256)
    print("out shape", out.shape)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2)
    ax[0].imshow(x.squeeze(), cmap="gray")
    ax[1].imshow(out.detach().squeeze(), cmap="gray")
    plt.show()
    
