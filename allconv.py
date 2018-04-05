import torch.nn as nn

class AllConvNet(nn.Module):
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(AllConvNet, self).__init__()
        self.input_drop = nn.Dropout(.2)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_size, 96, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(96, 96, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Dropout(.5),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(96, 192, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(192, 192, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(192, 192, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Dropout(.5),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(192, 192, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(192, 192, 1),
            nn.ReLU(),
        )
        self.class_conv = nn.Conv2d(192, n_classes, 1)
        self.global_avg = nn.AdaptiveAvgPool2d(1)


    def forward(self, x):
        x = self.input_drop(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.class_conv(x)
        x = self.global_avg(x)
        return x.squeeze()
