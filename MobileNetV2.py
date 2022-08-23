import torch 
from torch import nn 
MobileNet=[(16,1),(24,2),(24,1),(32,2),(32,1),(32,1),(64,2),(64,1),(64,1),(64,1),(96,1),(96,1),(96,1),(160,2),(160,1),(320,1)]

class BottleNeck(nn.Module):
    def __init__(self,in_channel,out_channel,t,stride):
        super(BottleNeck,self).__init__()
        self.in_channel=in_channel
        self.out_channel=out_channel
        self.stride=stride
        self.stack=nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=in_channel*t,kernel_size=1),
            nn.ReLU6(),
            nn.Conv2d(in_channels=in_channel*t,out_channels=in_channel*t,kernel_size=3,stride=stride,padding=1,groups=in_channel*t),
            nn.ReLU6(),
            nn.Conv2d(in_channels=in_channel*t,out_channels=out_channel,kernel_size=1)
        )
        self.downsample=nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,stride=stride,kernel_size=3,padding=1),
            nn.ReLU6(inplace=False)
        )

    def forward(self,x):
        identity=x.clone()
        out=self.stack(x)
        identity=self.downsample(x)
        out+=identity.clone()

        return out

class MobileNetv2(nn.Module):
    def __init__(self,in_channel=3,num_classes=1000):
        super(MobileNetv2,self).__init__()
        self.in_channel=in_channel
        self.begin=nn.Conv2d(in_channels=in_channel,out_channels=32,stride=2,kernel_size=3,padding=1)
        self.NN=self.make_layers(MobileNet,32)
        self.end=nn.Sequential(
            nn.Conv2d(in_channels=320,out_channels=1280,kernel_size=1),
            nn.AvgPool2d(kernel_size=7),
            nn.Conv2d(kernel_size=1,in_channels=1280,out_channels=num_classes)
        )

    def forward(self,x):
        x=self.begin(x)
        x=self.NN(x)
        x=self.end(x)
        s=x.shape[0]
        out=x.view(s,-1)
        return out

    def make_layers(self,architecture,in_channel):
        layers=[]
        for (x,stride) in architecture:
            t=6
            if(x==16):
                t=1
            layers.append(BottleNeck(in_channel=in_channel,out_channel=x,stride=stride,t=t))
            in_channel=x

        return nn.Sequential(*layers)