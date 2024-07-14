import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channel=6, num_blocks=5, hidden_channel=64):
        super().__init__()

        model = [nn.Conv2d(in_channel, hidden_channel, kernel_size=4, stride=2, padding=1, bias=False),
                 nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(num_blocks-3):  
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            model += [
                nn.Conv2d(hidden_channel * nf_mult_prev, hidden_channel * nf_mult, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(hidden_channel * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n, 8)
        model += [
                nn.Conv2d(hidden_channel * nf_mult_prev, hidden_channel * nf_mult, kernel_size=4, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(hidden_channel * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        model += [nn.Conv2d(hidden_channel * nf_mult, 1, kernel_size=4, stride=1, padding=1, bias=False)]

        self.model = nn.Sequential(*model)

    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))
    
def test():
    x = torch.randn((1, 3, 256,256))
    y = torch.randn((1, 3, 256,256))
    dis = Discriminator(in_channel=6, num_blocks=5)
    print(dis(x, y).shape)

if __name__ == "__main__":
    test()

