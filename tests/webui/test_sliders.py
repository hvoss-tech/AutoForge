import numpy as np
import torch

from autoforge.webui.helpers.sliders import (
    derive_sliders_from_optimizer,
    derive_sliders_from_result,
)


class _FakeOptimizer:
    def __init__(self, disc_global, height_map):
        self._dg = torch.tensor(disc_global, dtype=torch.long)
        self._dh = torch.tensor(height_map, dtype=torch.float32)
        self.material_colors = torch.zeros((3, 3), dtype=torch.float32)

    def get_discretized_solution(self, best=True):
        return self._dg, self._dh


def test_derive_sliders_basic_stack():
    # Stack of 8 layers: material 0 for layers 1-2, material 1 for 3-5,
    # material 2 for 6-8.  Pixel heights range 1..8.
    opt = _FakeOptimizer([0, 0, 1, 1, 1, 2, 2, 2], np.array([[2, 8], [5, 3]]))
    tds = np.array([1.0, 2.0, 3.0])
    uuids = ["u0", "u1", "u2"]

    result = derive_sliders_from_optimizer(opt, tds, uuids, layer_height=0.04)

    assert result is not None
    assert result["min_layer"] == 2
    assert result["max_layer"] == 8
    assert len(result["sliders"]) == 3
    layers = [s["layer"] for s in result["sliders"]]
    assert layers == [2, 5, 8]
    assert [s["td"] for s in result["sliders"]] == [1.0, 2.0, 3.0]
    assert [s["filament_uuid"] for s in result["sliders"]] == ["u0", "u1", "u2"]
    assert all(s["enabled"] for s in result["sliders"])
    assert result["sliders"][2]["depth_mm"] == 0.32


def test_derive_sliders_is_not_capped():
    # Alternating materials across 40 layers -> 40 segments. Capping/merging
    # them would change what render_with_sliders reconstructs (see the
    # module docstring), so every band is kept.
    stack = [i % 2 for i in range(40)]
    opt = _FakeOptimizer(stack, np.full((4, 4), 40))
    tds = np.array([1.0, 2.0])
    uuids = ["u0", "u1"]

    result = derive_sliders_from_optimizer(opt, tds, uuids, layer_height=0.04)
    assert result is not None
    assert result["max_layer"] == 40
    assert len(result["sliders"]) == 40
    assert [s["filament_uuid"] for s in result["sliders"][:3]] == ["u0", "u1", "u0"]
    # layers stay strictly ascending and within range
    layers = [s["layer"] for s in result["sliders"]]
    assert layers == sorted(layers)
    assert 1 <= layers[0] and layers[-1] == 40


def test_derive_sliders_no_print():
    opt = _FakeOptimizer([0, 1, 1], np.zeros((4, 4)))
    result = derive_sliders_from_optimizer(opt, np.array([1.0, 2.0]), ["u0", "u1"], 0.04)
    assert result == {"sliders": [], "min_layer": 0, "max_layer": 0}


def test_derive_sliders_from_result():
    opt = _FakeOptimizer([0, 1], np.array([[1, 2]]))
    result = derive_sliders_from_result(
        {
            "optimizer": opt,
            "material_TDs_np": np.array([1.5, 2.5]),
            "material_uuids": ["ua", "ub"],
            "args": type("Args", (), {"layer_height": 0.08})(),
        }
    )
    assert result is not None
    assert len(result["sliders"]) == 2
    assert result["sliders"][1]["layer"] == 2
    assert result["sliders"][1]["td"] == 2.5
    assert result["sliders"][1]["filament_uuid"] == "ub"
    assert result["sliders"][1]["depth_mm"] == 0.16
