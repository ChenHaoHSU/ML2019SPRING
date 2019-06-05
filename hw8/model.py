import torch
import torch.nn as nn

class MobileNet_Li28(nn.Module):
    def __init__(self):
        super(MobileNet_Li28, self).__init__()
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn( 1, 8, 1), 
            conv_dw( 8, 8, 1),

            conv_bn( 8, 16, 1),
            conv_dw(16, 16, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            conv_bn(16, 32, 1),
            conv_dw(32, 32, 1),
            conv_dw(32, 32, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            conv_bn(32, 64, 1),
            conv_dw(64, 64, 1),
            conv_dw(64, 64, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            
            nn.AdaptiveAvgPool2d(output_size=(6,6))
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(2304, 28),
            #nn.Dropout(0.1),
            nn.BatchNorm1d(28),
            nn.ReLU(inplace=True),
            nn.Linear(28, 8),
        )
    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 64*6*6)
        out = self.fc(x)
        return out