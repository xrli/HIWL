import torch.nn as nn
import torch
#源tf码中全局池化前有bn，不同深度的先对输入进行bn-relu再变成shortcut，同深度shortcut直接对输入下采样（maxpooling k=1*1 strid=s）

class BasicBlock(nn.Module):

    def __init__(self, m, k=2, dropoutrate=0.2, istop : bool = False,isbottom : bool = False):
        super(BasicBlock, self).__init__()
        self.in_channel = m*k * 2
        self.out_channel = m*k * 4
        self.istop=istop
        self.isbottom=isbottom
        if self.istop:
            self.in_channel = 64
        self.conv1 = nn.Conv2d(in_channels=self.in_channel, out_channels=m*k, kernel_size=1,
                               stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=m*k, out_channels=m*k, kernel_size=3,
                               stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=m*k, out_channels=self.out_channel, kernel_size=1,
                               stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=self.out_channel, out_channels=m*k, kernel_size=1,
                               stride=1, padding=0)
        self.conv5 = nn.Conv2d(in_channels=m*k, out_channels=m*k, kernel_size=3,
                               stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=m*k, out_channels=self.out_channel, kernel_size=1,
                               stride=2, padding=0)

        self.convshortcut1= nn.Conv2d(in_channels=self.in_channel, out_channels=self.out_channel, kernel_size=1,#raise dimension
                               padding=0, stride=1)
        self.convshortcut2 = nn.MaxPool2d(kernel_size=2,stride=2)#downsample
        self.bninc = nn.BatchNorm2d(self.in_channel)
        self.bnmk = nn.BatchNorm2d(m*k)
        self.bnoutc = nn.BatchNorm2d(self.out_channel)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropoutrate)
        if self.isbottom:
            self.conv6 = nn.Conv2d(in_channels=m * k, out_channels=self.out_channel, kernel_size=1,
                                   stride=1, padding=0)


    def forward(self, x):
        #第一个块

        # identity1 = self.bninc(x)

        out = self.bninc(x)
        out = self.relu(out)
        identity1 = out
        out = self.conv1(out)
        out = self.bnmk(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.dropout(out)
        out = self.bnmk(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += self.convshortcut1(identity1)
        #第二个块
        # identity2 = self.bnoutc(out)
        identity2 = out
        out = self.bnoutc(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.bnmk(out)
        out = self.relu(out)
        out = self.conv5(out)
        out = self.dropout(out)
        out = self.bnmk(out)
        out = self.relu(out)
        out = self.conv6(out)
        if self.isbottom:
            out += identity2
            out = self.bnoutc(out)
        else:
            out += self.convshortcut2(identity2)

        return out

class ResNet26(nn.Module):

    def __init__(self,
                 block,
                 mlist,
                 # mlist=[32, 64, 128, 256],
                 k,
                 dropoutrate,
                 num_classes
                 ):
        super(ResNet26, self).__init__()

        self.pad=nn.ZeroPad2d(padding=(2, 3, 2, 3))
        self.conv1x = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=6, stride=1)
        self.maxpool = nn.MaxPool2d(kernel_size=1, stride=2)
        self.conv2to5x = self._make_layer(block, mlist, k, dropoutrate)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(mlist[-1]*k*4, num_classes)
        self.fc = nn.Sequential(
            # nn.Dropout(p=0.3),

            nn.Conv2d(in_channels=mlist[-1]*k*4,out_channels=num_classes, kernel_size=1)
        )
    def forward(self, x):
        out = self.pad(x)
        out = self.conv1x(out)
        out = self.maxpool(out)
        out = self.conv2to5x(out)
        out = self.avgpool(out)
        # out = torch.flatten(out, start_dim=1)
        out = self.fc(out)
        out = torch.flatten(out, start_dim=1, end_dim=3)
        return out

    def _make_layer(self, block, mlist, k, dropoutrate):
        layers = []
        for i in range(len(mlist)):
            if i == 0:
                layers.append(block(m=mlist[i], k=k, dropoutrate=dropoutrate, istop= True, isbottom=False))
            elif (i == len(mlist)-1):
                layers.append(block(m=mlist[i], k=k, dropoutrate=dropoutrate, istop=False, isbottom=True))
            else:
                layers.append(block(m=mlist[i], k=k, dropoutrate=dropoutrate, istop=False, isbottom=False))
        return nn.Sequential(*layers)
def resnet26(block=BasicBlock, mlist=[64, 128, 256, 512], k=2, dropoutrate=0.33, num_classes=5):
    return ResNet26(block=block, mlist=mlist,  k=k, dropoutrate=dropoutrate, num_classes=num_classes)
# from torchsummary import summary
# summary(resnet26().cuda(),(3,64,64))
