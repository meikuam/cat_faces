import torch.nn as nn


class ModelWrapper(nn.Module):

    def __init__(self, base_model: nn.Module):
        super(ModelWrapper, self).__init__()

        self.base_model = base_model
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input_shape = x.shape[2:]

        y = self.base_model(x)
        y = nn.functional.interpolate(y, size=input_shape, mode='bilinear', align_corners=False)
        y = self.sigmoid(y)
        return y
