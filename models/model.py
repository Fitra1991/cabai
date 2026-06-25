import torch
import torch.nn as nn
from torchvision import models


class MultimodalDenseNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        # weights=None supaya Streamlit Cloud tidak mencoba download pretrained weights.
        self.rgb_net = models.densenet121(weights=None)
        rgb_feat = self.rgb_net.classifier.in_features
        self.rgb_net.classifier = nn.Identity()

        self.th_net = models.densenet121(weights=None)
        th_feat = self.th_net.classifier.in_features
        self.th_net.classifier = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(rgb_feat + th_feat, num_classes)
        )

    def forward(self, rgb, th):
        rgb_feat = self.rgb_net(rgb)
        th_feat = self.th_net(th)
        fused = torch.cat([rgb_feat, th_feat], dim=1)
        return self.classifier(fused)
