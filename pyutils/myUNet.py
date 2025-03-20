import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class DoubleConv(nn.Module):
    """(Convolution => BatchNorm => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        
        self.encoder = nn.ModuleList([
            DoubleConv(in_channels, 64),
            DoubleConv(64, 128),
            DoubleConv(128, 256),
            DoubleConv(256, 512),
        ])
        
        self.pool = nn.MaxPool2d(2)
        
        self.bottleneck = DoubleConv(512, 1024)
        
        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
        ])
        
        self.decoder = nn.ModuleList([
            DoubleConv(1024, 512),
            DoubleConv(512, 256),
            DoubleConv(256, 128),
            DoubleConv(128, 64),
        ])
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        input_dim = x.dim()
        # Ensure input has a channel dimension
        x = x.unsqueeze(1) if input_dim == 3 else x
        
        skip_connections = []
        
        # Encoder
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        skip_connections = skip_connections[::-1]
        for idx in range(len(self.decoder)):
            x = self.upconvs[idx](x)
            skip_connection = skip_connections[idx]
            
            # Concatenation with skip connection
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)
            
            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx](x)
        
        x = self.final_conv(x)
        
        # Remove channel dimension if necessary
        return x.squeeze(1) if input_dim == 3 else x

# Example usage
if __name__ == "__main__":
    model = UNet(in_channels=1, out_channels=1)
    x = torch.randn((1, 1, 320, 320))  # Example input tensor (Batch, Width, Height)
    t = time.time()
    out = model(x)


    print("time =", time.time() - t)
    print("in shape", out.shape)  # Should be (1, 256, 256)
    print("out shape", x.shape)
    import matplotlib.pyplot as plt
    plt.imshow(out.detach().numpy(). squeeze() , cmap="gray")
    plt.show()
