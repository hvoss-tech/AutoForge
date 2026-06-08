import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import VGG16_Weights


class MultiLayerVGGPerceptualLoss(nn.Module):
    def __init__(self, layers: list = None, weights: list = None):
        """
        Uses a pretrained VGG16 model to extract features from multiple layers.
        By default, it uses layers [3, 8, 15, 22] (approximately conv1_2, conv2_2, conv3_3, conv4_3).
        """
        super(MultiLayerVGGPerceptualLoss, self).__init__()
        # Choose layers from VGG16.features
        if layers is None:
            layers = [
                8,
            ]
        self.layers = layers

        # Load pretrained VGG16 and freeze parameters.
        vgg = models.vgg16(weights=VGG16_Weights.DEFAULT).features
        for param in vgg.parameters():
            param.requires_grad = False

        # We want to run the network up to the highest required layer.
        self.vgg = nn.Sequential(*[vgg[i] for i in range(max(layers) + 1)]).eval()

        # Weights for each selected layer loss; default: equal weighting.
        if weights is None:
            weights = [1.0 / len(layers)] * len(layers)
        self.weights = weights

        # Register ImageNet normalization constants as buffers.
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x and y are expected to be of shape (N, 3, H, W) with values in [0, 255].
        They are normalized to ImageNet stats and then passed through VGG16.
        The loss is computed as a weighted sum of MSE losses on the selected layers.
        """
        # Normalize images
        x = (x / 255.0 - self.mean) / self.std
        y = (y / 255.0 - self.mean) / self.std

        with torch.no_grad():
            y_features = {}
            y_out = y
            for i, layer in enumerate(self.vgg):
                y_out = layer(y_out)
                if i in self.layers:
                    y_features[i] = y_out

        loss = 0.0
        out = x
        for i, layer in enumerate(self.vgg):
            out = layer(out)
            if i in self.layers:
                loss += self.weights[self.layers.index(i)] * F.mse_loss(out, y_features[i])
        return loss
