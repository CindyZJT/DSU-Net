from fpn3d.resnet import ResNet, Bottleneck, BasicBlock
import torch
import importlib

from torch import nn
from torch.nn import functional as F



class FPN3D(nn.Module):
    def __init__(self,**kwargs):
        super(FPN3D,self).__init__()

        resnet = ResNet(BasicBlock, [3, 4, 6, 3])

        self.layer0 = nn.Conv3d(1,64,kernel_size=1,bias=False)

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.down4 = nn.Sequential(
            nn.Conv3d(512,32,kernel_size=1,bias=False),
            nn.GroupNorm(num_groups=8,num_channels=32),
            nn.ReLU()
        )

        self.down3 = nn.Sequential(
            nn.Conv3d(256, 32, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.ReLU()
        )

        self.down2 = nn.Sequential(
            nn.Conv3d(128, 32, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.ReLU()
        )

        self.down1 = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.ReLU()
        )

        self.predict = nn.Sequential(
            nn.Conv3d(32,8,kernel_size=3,padding=1,bias=False),
            nn.GroupNorm(num_groups=8,num_channels=8),
            nn.ReLU(),
            nn.Dropout3d(0.1),
            nn.Conv3d(8,1,kernel_size=1)
        )


        for m in self.modules():
            if isinstance(m,nn.ReLU) or isinstance(m, nn.Dropout3d):
                m_inplace = True

    def forward(self,x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        down4 = self.down4(layer4)
        down3 = self.down3(layer3)
        down2 = self.down2(layer2)
        down1 = self.down1(layer1)

        down4 = F.upsample(down4,size=down3.size()[2:],mode='trilinear')
        refine3 = down4+down3
        refine3 = F.upsample(refine3,size=down2.size()[2:],mode='trilinear')
        refine2 = refine3 + down2
        refine2 = F.upsample(refine2,size=down1.size()[2:],mode='trilinear')
        refine1 = refine2 + down1

        refine4 = F.upsample(down4,size=down1.size()[2:],mode='trilinear')
        refine3 = F.upsample(refine3,size=down1.size()[2:],mode='trilinear')
        refine2 = F.upsample(refine2,size=down1.size()[2:],mode='trilinear')

        predict4 = self.predict(refine4)
        predict3 = self.predict(refine3)
        predict2 = self.predict(refine2)
        predict1 = self.predict(refine1)

        predict4 = F.upsample(predict4,size=x.size()[2:],mode='trilinear')
        predict3 = F.upsample(predict3,size=x.size()[2:],mode='trilinear')
        predict2 = F.upsample(predict2,size=x.size()[2:],mode='trilinear')
        predict1 = F.upsample(predict1,size=x.size()[2:],mode='trilinear')

        if not self.training:
            return F.sigmoid(predict1)

        return predict1,predict2,predict3,predict4



def get_model_fpn(config):
    def _model_class(class_name):
        m = importlib.import_module('fpn3d.model')
        clazz = getattr(m, class_name)
        return clazz

    assert 'model' in config, 'Could not find model configuration'
    model_config = config['model']
    model_class = _model_class(model_config['name'])

    return model_class(**model_config)