import torch
from pytorch.model import UNet
H = W = 128
channels = 3

def main():
    inp = torch.randn(1, channels, H, W)
    model = UNet(inc=channels, num_cls=6)
    x = model(inp)
    print(x.shape)

if __name__=="__main__":
    main()
