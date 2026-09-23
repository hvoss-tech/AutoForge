"""Tests for autoforge.Loss.PerceptionLoss.MultiLayerVGGPerceptualLoss.

The VGG16 backbone is monkeypatched to a tiny stand-in so no pretrained
weights are downloaded.
"""

import pytest
import torch

from autoforge.Loss.PerceptionLoss import MultiLayerVGGPerceptualLoss


@pytest.fixture(autouse=True)
def mock_vgg16(monkeypatch):
    import torchvision.models as models

    class DummyFeatures(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList(
                [torch.nn.Conv2d(3, 3, 3, padding=1) if i % 2 == 0 else torch.nn.ReLU()
                 for i in range(9)]
            )

        def __getitem__(self, idx):
            return self.layers[idx]

        def __iter__(self):
            return iter(self.layers)

        def __len__(self):
            return len(self.layers)

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = DummyFeatures()

    monkeypatch.setattr(models, "vgg16", lambda weights=None: DummyModel())
    yield


def test_identical_inputs_give_zero_loss():
    loss_mod = MultiLayerVGGPerceptualLoss(layers=[8])
    x = torch.rand(1, 3, 32, 32) * 255.0
    out = loss_mod(x, x.clone())
    assert torch.isfinite(out)
    assert out.item() == pytest.approx(0.0, abs=1e-5)


def test_different_inputs_give_positive_loss():
    loss_mod = MultiLayerVGGPerceptualLoss(layers=[8])
    x = torch.rand(1, 3, 32, 32) * 255.0
    y = torch.rand(1, 3, 32, 32) * 255.0
    out = loss_mod(x, y)
    assert out.item() > 0.0


def test_loss_is_differentiable_wrt_input():
    loss_mod = MultiLayerVGGPerceptualLoss(layers=[4, 8])
    x = (torch.rand(1, 3, 16, 16) * 255.0).requires_grad_(True)
    y = torch.rand(1, 3, 16, 16) * 255.0
    loss_mod(x, y).backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
