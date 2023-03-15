import torch
import torch.nn as nn
from torchvision import models


class Res50(nn.Module):
    def __init__(self, num_class: int, checkpoint: str = '', bottleneck: bool = True, bottleneck_dim: int = 256):
        super(Res50, self).__init__()
        self.num_class = num_class
        self.checkpoint = checkpoint
        self.is_bottle = bottleneck
        self.feat_dim = bottleneck_dim

        # load checkpoint
        if checkpoint:
            self.model = models.resnet50(weights=None)
            checkpoints = torch.load(checkpoint)
            state_dict = checkpoints['state_dict']
            for k in list(state_dict.keys()):

                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):

                    # remove prefix
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]

                # delete renamed or unused k
                del state_dict[k]
            self.model.load_state_dict(state_dict, strict=False)
        else:
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        out_dim = self.model.fc.in_features

        self.bottleneck = nn.Sequential(
            nn.Linear(out_dim, self.feat_dim),
            nn.BatchNorm1d(self.feat_dim),
            nn.ReLU()
        )

        if self.is_bottle:
            self.head = nn.Linear(self.feat_dim, self.num_class)
        else:
            self.head = nn.Linear(2048, self.num_class)

    def forward(self, x):

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.is_bottle:
            feature = self.bottleneck(x)
            prediction = self.head(feature)
            return prediction, feature
        else:
            prediction = self.head(x)
            return prediction, x

    def get_parameters(self, base_lr=1.0):
        if self.is_bottle:
            params = [
                {"params": self.model.parameters(), "lr": 0.1 * base_lr},
                {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
                {"params": self.head.parameters(), "lr": 1.0 * base_lr}
            ]
        else:
            params = [
                {"params": self.model.parameters(), "lr": 0.1 * base_lr},
                {"params": self.head.parameters(), "lr": 1.0 * base_lr}
            ]
        return params

    def load_model_state_dict(self, state_dict):
        for k in list(state_dict.keys()):

            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]

            # delete renamed or unused k
            del state_dict[k]
        self.model.load_state_dict(state_dict, strict=False)


