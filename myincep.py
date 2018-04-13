import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.inception import InceptionAux

model = models.inception_v3()
model.fc = torch.nn.Linear(2048, 26)
model.aux_logits = InceptionAux(768, 26)

model.load_state_dict(torch.load('Best_inception_v3_fl_enhanced.pth.tar')['state_dict'])

class MyInception(nn.Module):
    def __init__(self, num_classes, aux_logits=True):
        super(MyInception, self).__init__()
        self.aux_logits = aux_logits
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        if aux_logits:
            self.AuxLogits = MyInceptionAux(num_classes)
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c
        self.fc = nn.Linear(2048, num_classes)
    def forward(self, x):
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = torch.nn.functional.avg_pool2d(x, kernel_size=8)
        x = torch.nn.functional.dropout(x, training=self.training)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.training and self.aux_logits:
            return x, aux
        return x

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return torch.nn.functional.relu(x, inplace=True)

class MyInceptionAux(nn.Module):
    def __init__(self, num_classes):
        super(MyInceptionAux, self).__init__()
        self.conv0 = BasicConv2d(768, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(12288, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        # 17 x 17 x 768
        x = nn.functional.avg_pool2d(x, kernel_size=5, stride=3)
        # 5 x 5 x 768
        x = self.conv0(x)
        # 5 x 5 x 128
        x = self.conv1(x)
        # 1 x 1 x 768
        x = x.view(x.size(0), -1)
        # 768
        x = self.fc(x)
        # 1000
        return x