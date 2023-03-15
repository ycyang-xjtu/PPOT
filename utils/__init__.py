import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from modules.resnet import Res50
import numpy as np


def get_prototypes(feature_set: torch.Tensor, label_set: torch.Tensor, args) -> torch.Tensor:
    class_set = [i for i in range(args.common_class + args.source_private_class)]
    source_prototype = torch.zeros(len(class_set), 256)
    for i in class_set:
        source_prototype[i] = feature_set[label_set == i].sum(0) / feature_set[label_set == i].size(0)
    return source_prototype


def h_score(acc_known: float, acc_unknown: float) -> float:
    h_scores = 2 * acc_known * acc_unknown / (acc_known + acc_unknown)
    return h_scores


def entropy_loss(prediction: torch.Tensor, weight=torch.zeros(1)):
    if weight.size(0) == 1:
        entropy = torch.sum(-prediction * torch.log(prediction + 1e-8), 1)
        entropy = torch.mean(entropy)
    else:
        entropy = torch.sum(-prediction * torch.log(prediction + 1e-8), 1)
        entropy = torch.mean(weight * entropy)
    return entropy