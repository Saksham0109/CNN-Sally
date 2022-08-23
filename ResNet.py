import torch 
from torch import nn 
ResNet34=[64,64,64,128,128,128,128,256,256,256,256,256,256,512,512,512]

class Block(nn.Module):
    def __init__(self,in_channel,out_channel,stride):
        super(Block,self).__init__()
        self.in_channel=in_channel
        self.out_channel=out_channel
        self.stride=stride
        self.stack=nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,stride=stride,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel,out_channels=out_channel,kernel_size=3,padding=1),
            nn.ReLU()
        )
        if(stride==2):
            self.downsample=nn.Sequential(
                nn.Conv2d(in_channels=in_channel,out_channels=out_channel,stride=stride,kernel_size=3,padding=1),
                nn.ReLU(inplace=False)
            )

    def forward(self,x):
        identity=x.clone()
        out=self.stack(x)
        if(self.stride==2):
            identity=self.downsample(x)
        out+=identity.clone()

        return out

class ResNet(nn.Module):
    def __init__(self,in_channel=3,num_classes=1000):
        super(ResNet,self).__init__()
        self.in_channel=in_channel
        self.begin=nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=64,stride=2,kernel_size=7,padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.NN=self.make_layers(ResNet34,64)
        self.fc=nn.Linear(512*7*7,num_classes)

    def forward(self,x):
        x=self.begin(x)
        x=self.NN(x)
        s=x.shape[0]
        out=x.view(s,-1)
        out=self.fc(out)
        return out

    def make_layers(self,architecture,in_channel):
        layers=[]
        for x in architecture:
            stride=1
            if(x!=in_channel):
                stride=2
            layers.append(Block(in_channel=in_channel,out_channel=x,stride=stride))
            in_channel=x

        return nn.Sequential(*layers)