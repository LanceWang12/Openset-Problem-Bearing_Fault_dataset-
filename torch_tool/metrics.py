import torch
from torch import nn

def categorical_accuracy(y_true, y_pred):
    result = torch.argmax(y_pred, dim = 1)
    correct = torch.sum((y_true == result))
    acc = correct / len(y_true)
    return acc

def binary_accuracy(y_true, y_pred):
    y_true = torch.as_tensor(y_true)
    result = (nn.Sigmoid()(y_pred) > 0.5)
    correct = torch.sum(result == y_true)
    acc = correct / len(result)
    return acc