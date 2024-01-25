import torch.nn as nn
from functions import ReverseLayerF,CenterLoss
import model.back_bone as backbone

class CDANet(nn.Module):
    def __init__(self,number_classes=31,base_net='ResNet50'):
        super(CDANet,self).__init__()
        self.sharedNet = backbone.network_dict[base_net]()
        self.bottleneck = nn.Linear(2048, 256)
        self.source_fc = nn.Linear(256, number_classes)
        self.softmax = nn.Softmax(dim=1)
        self.classes = number_classes


    def forward(self, source, target, alpha = 0.0):
        source_share = self.sharedNet(source)
        source_share = self.bottleneck(source_share)
        source = self.source_fc(source_share)

        target_share = self.sharedNet(target)
        target_share = self.bottleneck(target_share)
        target = self.source_fc(target_share)

        return source, target, source_share, target_share
