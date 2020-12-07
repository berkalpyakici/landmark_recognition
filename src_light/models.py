import timm
from torch import nn

import math
import torch
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from efficientnet_pytorch import EfficientNet
import geffnet

import pytorch_lightning as pl

class ArcMarginProduct(pl.LightningModule):
    def __init__(self, in_features, data_module):
        super().__init__()
        print("TESTING HERE:",data_module.get_dim())
        self.weight = nn.Parameter(torch.Tensor(data_module.get_dim(), in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine

class ArcFaceLoss(pl.LightningModule):
    def __init__(self, s=45.0, m=0.35):
        super().__init__()

        self.crit = nn.CrossEntropyLoss(reduction="mean")   
        self.s = s

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        
    def forward(self, logits, labels):
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        labels2 = torch.zeros_like(cosine)
        labels2.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (labels2 * phi) + ((1.0 - labels2) * cosine)

        output = output * self.s
        loss = self.crit(output, labels)
        return loss

class Effnet_Landmark(pl.LightningModule):
    def __init__(self, args, data_module):
        super().__init__()

        #self.effnet = timm.create_model('tf_efficientnet_b7', pretrained=True)
        self.effnet = geffnet.create_model('tf_efficientnet_b7_ns', pretrained = True)
        self.in_features = self.effnet.classifier.in_features
        self.effnet.classifier = nn.Identity()

        self.embedding_size = 512

        self.global_pool = nn.Identity() #placeholder if we want to add pooling
        self.neck = nn.Sequential(
            nn.Linear(self.in_features, self.embedding_size),
            nn.BatchNorm1d(self.embedding_size),
            nn.ReLU()
        )

        self.head = ArcMarginProduct(self.embedding_size, data_module)
    
    def forward(self, x):
        x = self.effnet(x)
        #x = self.global_pool(x)
        x = self.neck(x)

        return self.head(x)