import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    def __init__(self, in_channel, out_channel, submodule=None, inner_most=False, outer_most=False, use_dropout=False):
        super().__init__()

        downconv = nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1, bias=False)
        downnorm = nn.BatchNorm2d(out_channel)
        downrelu = nn.LeakyReLU(0.2, True)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(in_channel)

        self.outer_most = outer_most
        
        if inner_most:
            upconv = nn.ConvTranspose2d(out_channel, in_channel, kernel_size=4, stride=2, padding=1, bias=False)
            model = [downconv, downrelu, upconv, upnorm, uprelu]
        elif outer_most:
            upconv = nn.ConvTranspose2d(out_channel*2, in_channel, kernel_size=4, stride=2, padding=1, bias=False)
            model = [downconv, submodule, upconv, nn.Tanh()]
        else:
            upconv = nn.ConvTranspose2d(out_channel*2, in_channel, kernel_size=4, stride=2, padding=1, bias=False)
            model = [downconv, downnorm, downrelu, submodule, upconv, upnorm, uprelu]
            if use_dropout:
                model += [nn.Dropout(0.5)]
        
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outer_most:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], dim=1)
        
class Generator(nn.Module):
    def __init__(self, in_channel=3, hidden_channel=64, num_blocks=8, use_dropout=True):
        super().__init__()
        
        block = UNetBlock(hidden_channel*8, hidden_channel*8, inner_most=True)
        for _ in range(num_blocks-5):
            block = UNetBlock(hidden_channel*8, hidden_channel*8, submodule=block, use_dropout=use_dropout)
        block = UNetBlock(hidden_channel*4, hidden_channel*8, submodule=block, use_dropout=use_dropout)
        block = UNetBlock(hidden_channel*2, hidden_channel*4, submodule=block, use_dropout=use_dropout)
        block = UNetBlock(hidden_channel, hidden_channel*2, submodule=block, use_dropout=use_dropout)
        self.model = UNetBlock(in_channel, hidden_channel, submodule=block, outer_most=True)
    
    def forward(self, x):
        return self.model(x)

def test():
    x = torch.randn((1, 3, 256,256))
    gen = Generator(in_channel=3, hidden_channel=64)
    print(gen(x).shape)

if __name__ == "__main__":
    test()