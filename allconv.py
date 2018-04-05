import torch.nn as nn
import torch.nn.functional as F

class AllConvNet(nn.Module):
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(AllConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 1)

        self.class_conv = nn.Conv2d(192, n_classes, 1)
        self.global_avg = nn.AdaptiveAvgPool2d(1)


    def forward(self, x):
        x_drop = F.dropout(x, .2)
        conv1_out = F.relu(self.conv1(x_drop))
        conv2_out = F.relu(self.conv1(conv1_out))
        conv3_out = F.relu(self.conv1(conv2_out))
        conv3_out_drop = F.dropout(conv3_out, .5)
        conv4_out = F.relu(self.conv1(conv3_out_drop))
        conv5_out = F.relu(self.conv1(conv4_out))
        conv6_out = F.relu(self.conv1(conv5_out))
        conv6_out_drop = F.dropout(conv6_out, .5)
        conv7_out = F.relu(self.conv1(conv6_out_drop))
        conv8_out = F.relu(self.conv1(conv7_out))

        class_out = F.relu(self.class_conv(conv8_out))
        pool_out = self.adaptive_avg_pool2d(class_out, 1).squeeze()
        return pool_out
