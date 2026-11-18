"""Height guidance (Helper/HeightGuidance.py): filament pins, stack order and
pixel height targets as soft costs, and their use across training, the
discrete searches and pruning."""
import json

import numpy as np
import pytest
import torch

from autoforge.Helper.HeightAssign import assign_heights
from autoforge.Helper.HeightGuidance import (
    PIXEL_TOLERANCE,
    GuidanceSpec,
    HeightGuidance,
    build_target_maps,
    material_index_lookup,
    parse_spec,
)

# black, white, red, green, blue, yellow
COLORS = torch.tensor(
    [[0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 0.66, 0], [0, 0, 1], [1, 1, 0]], dtype=torch.float32
)


def _guidance(**kw):
    maps = kw.pop("maps", (None, None))
    return HeightGuidance(GuidanceSpec(**kw), COLORS, 20, *maps, device="cpu", layer_height=0.04)


# --------------------------------------------------------------------------
# spec


def test_parse_spec_reads_json_strings_files_and_resolves_filaments(tmp_path):
    lookup = material_index_lookup(["ua", "ub", "uc"], ["Black", "White", "Red"], ["#000000", "#ffffff", "#ff0000"])
    data = {
        "strength": 2,
        "order": "dark_to_light",
        "pins": [{"uuid": "uc", "position": 0.9}, {"filament": "white", "position": 0.1},
                 {"color": "#000000", "position": 1.4}, {"uuid": "gone", "position": 0.5}],
        "colors": [{"color": "#00ff00", "tolerance": 20, "direction": "lower"}],
    }
    path = tmp_path / "g.json"
    path.write_text(json.dumps(data))
    for source in (data, json.dumps(data), str(path)):
        spec = parse_spec(source, lookup)
        assert spec.strength == 2 and spec.order == "dark_to_light"
        # Unknown filaments are dropped; positions clamp to the print.
        assert spec.pins == [(2, 0.9), (1, 0.1), (0, 1.0)]
        assert spec.colors == [((0, 255, 0), 20.0, -1)]
    assert parse_spec(None) is None and parse_spec("") is None
    with pytest.raises(ValueError):
        parse_spec({"order": "sideways"})


# --------------------------------------------------------------------------
# stack cost


def _stack(*bands):
    """A 20-layer stack from (material, first layer, last layer) bands
    (1-based print layers); unlisted layers are blue (4)."""
    dg = torch.full((20,), 4, dtype=torch.long)
    for m, a, b in bands:
        dg[a - 1 : b] = m
    return dg


def test_a_pin_is_met_by_a_visible_band_near_it():
    g = _guidance(pins=[(2, 0.8)])  # red at layer 16 of 20
    near = _stack((2, 15, 17))
    assert float(g.stack_cost(near, base_material=0)) == pytest.approx(0.0, abs=1e-6)
    far = _stack((2, 2, 4))
    absent = _stack()
    assert float(g.stack_cost(absent, base_material=0)) == pytest.approx(g.stack_scale)
    assert 0.0 < float(g.stack_cost(far, base_material=0)) <= float(g.stack_cost(absent, base_material=0))
    # A single thin layer does not show; it counts for a third of a band.
    thin = _stack((2, 16, 16))
    assert 0.0 < float(g.stack_cost(thin, base_material=0)) < float(g.stack_cost(far, base_material=0))


def test_a_pin_only_needs_one_band_near_it():
    """The same filament elsewhere (colour mixing) costs nothing extra."""
    g = _guidance(pins=[(2, 0.8)])
    near = _stack((2, 15, 17))
    also_low = _stack((2, 2, 4), (2, 15, 17))
    assert float(g.stack_cost(also_low, base_material=0)) == pytest.approx(float(g.stack_cost(near, base_material=0)))


def test_a_pin_counts_only_layers_up_to_the_top():
    g = _guidance(pins=[(2, 0.8)])
    dg = _stack((2, 15, 17))
    g.set_top(10)  # the band sits above the printed top now
    assert float(g.stack_cost(dg, base_material=0)) == pytest.approx(g.stack_scale)
    # The pin moved with the top: 0.8 of 10 layers.
    assert float(g.stack_cost(_stack((2, 7, 9)), base_material=0)) < 0.01 * g.stack_scale


def test_a_pin_at_the_bottom_is_met_by_the_base_filament():
    g = _guidance(pins=[(2, 0.0)])
    assert float(g.stack_cost(_stack(), base_material=2)) == pytest.approx(0.0, abs=1e-6)
    assert float(g.stack_cost(_stack(), base_material=0)) > 0.0


def test_order_costs_a_reversed_stack_most_and_local_mixing_little():
    g = _guidance(order="dark_to_light")
    bg = COLORS[0]
    dark_to_light = _stack((0, 1, 5), (4, 6, 10), (2, 11, 15), (1, 16, 20))
    reversed_ = torch.flip(dark_to_light, [0])
    mixing = dark_to_light.clone()
    mixing[[9, 10]] = mixing[[10, 9]]  # one blue/red pair swapped
    c_ok = float(g.stack_cost(dark_to_light, None, bg))
    c_mix = float(g.stack_cost(mixing, None, bg))
    c_rev = float(g.stack_cost(reversed_, None, bg))
    assert c_ok == pytest.approx(0.0, abs=1e-6)
    assert 0.0 < c_mix < 0.1 * c_rev
    # The other way round for light to dark.
    g2 = _guidance(order="light_to_dark")
    assert float(g2.stack_cost(dark_to_light, None, COLORS[1])) > float(g2.stack_cost(reversed_, None, COLORS[1]))


def test_soft_cost_of_one_hot_logits_is_the_discrete_cost():
    g = _guidance(pins=[(2, 0.8), (5, 0.3)], order="dark_to_light")
    gen = torch.Generator().manual_seed(0)
    for _ in range(5):
        dg = torch.randint(0, 6, (20,), generator=gen)
        logits = torch.full((20, 6), -30.0)
        logits[torch.arange(20), dg] = 30.0
        soft = float(g.stack_cost_soft(logits, None, 1, COLORS[1]))
        hard = float(g.stack_cost(dg, 1, COLORS[1]))
        assert soft == pytest.approx(hard, rel=1e-4, abs=1e-4)


def test_soft_cost_pulls_the_logits_toward_the_pin():
    g = _guidance(pins=[(2, 0.8)])
    logits = torch.zeros(20, 6, requires_grad=True)
    g.stack_cost_soft(logits, None, 0, COLORS[0]).backward()
    # Raising red near layer 16 lowers the cost; far below it doesn't matter.
    assert float(logits.grad[15, 2]) < 0.0
    assert abs(float(logits.grad[1, 2])) < 1e-3 * abs(float(logits.grad[15, 2]))


def test_batched_stack_cost_matches_single_stacks():
    g = _guidance(pins=[(2, 0.8)], order="dark_to_light")
    stacks = torch.randint(0, 6, (7, 20), generator=torch.Generator().manual_seed(3))
    batch = g.stack_cost(stacks, 1, COLORS[1])
    for i in range(7):
        assert float(batch[i]) == pytest.approx(float(g.stack_cost(stacks[i], 1, COLORS[1])), rel=1e-5)
    # Per-candidate tops (layer removal): the same as setting each top.
    tops = torch.tensor([20.0, 15, 10, 20, 12, 18, 9])
    per = g.stack_cost(stacks, 1, COLORS[1], top=tops)
    for i in range(7):
        g.set_top(int(tops[i]))
        assert float(per[i]) == pytest.approx(float(g.stack_cost(stacks[i], 1, COLORS[1])), rel=1e-5)


# --------------------------------------------------------------------------
# pixel targets


def test_pixel_cost_is_free_within_the_tolerance_and_grows_beyond():
    target = np.full((4, 4), 0.5, np.float32)
    weight = np.ones((4, 4), np.float32)
    g = _guidance(maps=(target, weight))
    z = torch.full((4, 4), 10)  # 0.5 of 20
    assert float(g.pixel_cost_map(z).max()) == 0.0
    z_tol = torch.full((4, 4), int(20 * (0.5 + PIXEL_TOLERANCE)))
    assert float(g.pixel_cost_map(z_tol).max()) == pytest.approx(0.0, abs=1e-5)
    c1 = g.pixel_cost_map(torch.full((4, 4), 16))
    c2 = g.pixel_cost_map(torch.full((4, 4), 20))
    assert float(c2.min()) > float(c1.min()) > 0.0
    # The table holds the same costs for every height, at any size.
    table = g.pixel_cost_table((4, 4), 21)
    for k in (0, 10, 16, 20):
        torch.testing.assert_close(table[..., k], g.pixel_cost_map(torch.full((4, 4), k)))
    assert g.pixel_cost_table((2, 2), 21).shape == (2, 2, 21)


def test_unweighted_pixels_cost_nothing():
    target = np.ones((4, 4), np.float32)
    weight = np.zeros((4, 4), np.float32)
    weight[0, 0] = 1.0
    g = _guidance(maps=(target, weight))
    c = g.pixel_cost_map(torch.zeros(4, 4, dtype=torch.long))
    assert float(c[0, 0]) > 0 and float(c.sum()) == float(c[0, 0])


def test_target_maps_from_depth_regions_colours_and_alpha():
    img = np.zeros((10, 10, 3), np.uint8)
    img[:, 5:] = (0, 200, 0)
    depth = np.tile(np.linspace(0, 1, 10, dtype=np.float32), (10, 1))
    t, w = build_target_maps(GuidanceSpec(depth=True), img, depth_fn=lambda _: depth)
    np.testing.assert_allclose(t, depth, atol=1e-6)
    assert np.all(w == 1.0)
    t, _ = build_target_maps(GuidanceSpec(depth=True, depth_flip=True), img, depth_fn=lambda _: depth)
    np.testing.assert_allclose(t, 1.0 - depth, atol=1e-6)

    # Painted regions: red raises, blue lowers; elsewhere unguided.
    regions = np.zeros((10, 10, 4), np.uint8)
    regions[:3, :] = (255, 0, 0, 255)
    regions[7:, :] = (0, 0, 255, 255)
    t, w = build_target_maps(GuidanceSpec(), img, regions)
    assert np.all(t[:3] == 1.0) and np.all(t[7:] == 0.0)
    assert np.all(w[:3] == 1.0) and np.all(w[3:7] == 0.0)

    # A picked colour pulls the pixels of that colour.
    t, w = build_target_maps(GuidanceSpec(colors=[((0, 200, 0), 15.0, 1)]), img)
    assert np.all(w[:, 5:] == 1.0) and np.all(w[:, :5] == 0.0)
    assert np.all(t[:, 5:] == 1.0)

    # Transparent pixels are never pulled.
    alpha = np.full((10, 10), 255, np.uint8)
    alpha[:, :2] = 0
    _, w = build_target_maps(GuidanceSpec(depth=True), img, alpha=alpha, depth_fn=lambda _: depth)
    assert np.all(w[:, :2] == 0.0) and np.all(w[:, 2:] == 1.0)

    assert build_target_maps(GuidanceSpec(), img) == (None, None)


def test_height_assignment_follows_the_pixel_cost():
    """Two heights show the same colour: the guidance decides between them."""
    colors = torch.tensor([[0.0, 0, 0], [200, 50, 50], [200, 50, 50], [255, 255, 255]])
    target = torch.zeros(6, 6, 3)
    target[...] = torch.tensor([200.0, 50, 50])
    tmap = np.zeros((6, 6), np.float32)
    tmap[:, 3:] = 1.0
    g = HeightGuidance(GuidanceSpec(), COLORS, 3, tmap, np.ones((6, 6), np.float32), device="cpu")
    z = assign_heights(colors, target, None, smooth=0.0, extra_cost=g.pixel_cost_table((6, 6), 4))
    assert torch.all(z[:, :3] == 1) and torch.all(z[:, 3:] == 2)


# --------------------------------------------------------------------------
# in the pipeline


@pytest.fixture
def _inputs(tmp_path):
    import cv2

    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:, :16] = (30, 120, 220)
    img[:16, 16:] = (240, 240, 240)
    img[16:, 16:] = (20, 160, 40)
    img_path = tmp_path / "img.png"
    cv2.imwrite(str(img_path), img)
    colors = ["#000000", "#ffffff", "#ff0000", "#00aa00", "#0000ff", "#ffff00"]
    fil = [{"color": c, "td": 2.0, "name": f"T - {c}", "uuid": f"u{i}"} for i, c in enumerate(colors)]
    return img_path, fil


def _run(tmp_path, img_path, fil, guidance, **extra):
    from autoforge.webui.helpers.pipeline_runner import run_pipeline

    settings = dict(iterations=400, stl_output_size=20, num_init_rounds=1, discrete_check=20, random_seed=1,
                    max_layers=30, height_guidance=guidance)
    settings.update(extra)
    return run_pipeline(str(img_path), fil, str(tmp_path / "w"), settings)


def _pin_report(opt, guidance=None):
    g = guidance or opt.guidance
    dg, z = opt.get_discretized_solution(best=True)
    return g.report(dg, z, opt.base_material, opt.background)["pins"][0]


CAT = "webui/frontend/tests/fixtures/cat_128x72.png"
CAT_FILAMENTS = [
    {"color": c, "td": t, "name": c, "uuid": f"u{i}"}
    for i, (c, t) in enumerate(zip(
        ["#000000", "#ffffff", "#c00d1e", "#00ae42", "#0a2989", "#f4ee2a", "#e7ceb5"], [0.6, 5, 4, 2, 2, 6, 8]
    ))
]
BEIGE_HIGH = {"strength": 1, "pins": [{"uuid": "u6", "position": 0.9}]}


def _cat_run(tmp_path, guidance, **extra):
    from autoforge.webui.helpers.pipeline_runner import run_pipeline

    return run_pipeline(CAT, CAT_FILAMENTS, str(tmp_path / "cat"), dict(
        iterations=3000, stl_output_size=30, num_init_rounds=1, random_seed=1, max_layers=40,
        height_guidance=guidance, **extra))["optimizer"]


def test_webui_run_follows_a_pin(tmp_path):
    """Without guidance this picture has no beige near the top; a pin puts
    a band there (through training and the intermediate search)."""
    opt = _cat_run(tmp_path, BEIGE_HIGH)
    assert opt.guidance is not None and len(opt.guidance.pins) == 1
    assert _pin_report(opt)["met"]

    plain = _cat_run(tmp_path, None, default_height_guidance=False)
    assert plain.guidance is None
    assert not _pin_report(plain, opt.guidance)["met"]


def test_pruning_keeps_a_pinned_band(tmp_path):
    """Pruning under a colour limit scores every candidate with the
    guidance: the pinned band survives it, and is lost when the same
    solution is pruned without the guidance."""
    opt = _cat_run(tmp_path, BEIGE_HIGH)
    g = opt.guidance
    assert _pin_report(opt)["met"]
    snap = opt.solution_snapshot()
    top = g.top
    opt.prune(5, 10, 0, 40, search_seed=False, fine_tune_height=False)
    assert _pin_report(opt)["met"]

    opt.restore_solution_snapshot(snap)
    g.set_top(top)
    opt.guidance = None
    opt.prune(5, 10, 0, 40, search_seed=False, fine_tune_height=False)
    assert not _pin_report(opt, g)["met"]


def test_training_graph_is_captured_with_guidance(tmp_path, _inputs):
    """The soft stack cost runs inside the captured training step: it must
    not sync with the host (prod's backward did, and every step fell back
    to the 5x slower eager path)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA graphs need CUDA")
    from autoforge.webui.helpers.pipeline_runner import build_pipeline_state

    img_path, fil = _inputs
    state = build_pipeline_state(str(img_path), fil, str(tmp_path / "g"), dict(
        iterations=100, stl_output_size=20, num_init_rounds=1, random_seed=1, max_layers=30, cuda_graph=True,
        height_guidance={"strength": 1, "order": "dark_to_light", "pins": [{"uuid": "u5", "position": 0.9}]}))
    opt = state["optimizer"]
    assert opt.guidance is not None and opt.guidance.has_stack
    for _ in range(8):
        opt.step()
    assert opt._graph is not None
    opt.release_cuda_graph()


def test_depth_guidance_raises_the_near_side(tmp_path, _inputs, monkeypatch):
    """Unguided, the right half of this picture prints far higher than the
    left; with a depth estimate nearer on the left the gap shrinks."""
    import autoforge.Helper.HeightGuidance as hg

    img_path, fil = _inputs
    monkeypatch.setattr(hg, "estimate_depth", lambda img: np.tile(
        np.linspace(1, 0, img.shape[1], dtype=np.float32), (img.shape[0], 1)))
    guided = _run(tmp_path, img_path, fil, {"strength": 5, "depth": True})["optimizer"]
    plain = _run(tmp_path, img_path, fil, None, default_height_guidance=False)["optimizer"]

    def right_minus_left(opt):
        _, z = opt.get_discretized_solution(best=True)
        z = z.float() / z.max()
        return float(z[:, z.shape[1] // 2:].mean() - z[:, : z.shape[1] // 2].mean())

    assert guided.guidance.has_pixel
    assert right_minus_left(guided) < right_minus_left(plain) - 0.05


def test_no_guidance_attaches_nothing(tmp_path, _inputs):
    img_path, fil = _inputs
    for g in (None, {"strength": 0, "pins": [{"uuid": "u5", "position": 0.9}]}, {"strength": 1}):
        state = _run(tmp_path, img_path, fil, g, iterations=20, default_height_guidance=False)
        assert state["optimizer"].guidance is None


def _fake_depth(monkeypatch):
    import autoforge.Helper.HeightGuidance as hg

    monkeypatch.setattr(hg, "estimate_depth", lambda img: np.tile(
        np.linspace(1, 0, img.shape[1], dtype=np.float32), (img.shape[0], 1)))


def test_webui_runs_follow_the_depth_by_default_without_showing_it(tmp_path, _inputs, monkeypatch):
    """A webui run whose user set no height guidance follows the photo's
    depth at strength 1 in the background: never reported, and the user's
    own setting stays empty (it is what gets saved)."""
    from autoforge.webui.helpers.sliders import guidance_report_from_optimizer

    _fake_depth(monkeypatch)
    img_path, fil = _inputs
    for g in (None, {"strength": 3}):
        state = _run(tmp_path, img_path, fil, g, iterations=20)
        guidance = state["optimizer"].guidance
        assert guidance is not None and guidance.hidden
        assert guidance.has_pixel and guidance.spec.depth and guidance.spec.strength == 1.0
        assert guidance_report_from_optimizer(state["optimizer"]) is None
        assert state["args"].height_guidance == g


def test_any_height_setting_replaces_the_default(tmp_path, _inputs, monkeypatch):
    _fake_depth(monkeypatch)
    img_path, fil = _inputs
    # Strength 0 with depth on: the user switched it off.
    assert _run(tmp_path, img_path, fil, {"strength": 0, "depth": True}, iterations=20)["optimizer"].guidance is None
    pinned = _run(tmp_path, img_path, fil, {"strength": 1, "pins": [{"uuid": "u5", "position": 0.9}]}, iterations=20)
    g = pinned["optimizer"].guidance
    assert g is not None and not g.hidden and not g.has_pixel and len(g.pins) == 1


def test_default_guidance_falls_back_without_a_depth_model(tmp_path, _inputs, monkeypatch):
    """No working depth model (no 'transformers', no download): the run
    goes on unguided instead of failing."""
    import autoforge.Helper.HeightGuidance as hg

    def broken(img):
        raise ImportError("Depth guidance needs 'transformers' installed.")

    monkeypatch.setattr(hg, "estimate_depth", broken)
    img_path, fil = _inputs
    state = _run(tmp_path, img_path, fil, None, iterations=20)
    assert state["optimizer"].guidance is None
    # The user's own depth guidance still reports the missing model.
    with pytest.raises(ImportError):
        _run(tmp_path, img_path, fil, {"strength": 1, "depth": True}, iterations=20)


def test_cli_takes_height_guidance(monkeypatch, tmp_path, _inputs):
    import cv2
    from autoforge.auto_forge import parse_args, start

    img_path, _ = _inputs
    csv = tmp_path / "filaments.csv"
    rows = [("#000000", "Black"), ("#ffffff", "White"), ("#ff0000", "Red"), ("#00aa00", "Green"),
            ("#0000ff", "Blue"), ("#ffff00", "Yellow")]
    csv.write_text("Brand, Type, Color, Name, TD, Owned, Uuid\n"
                   + "".join(f"B,PLA,{c},{n},2.0,false,{{{i}}}\n" for i, (c, n) in enumerate(rows)))
    regions = np.zeros((32, 32, 4), np.uint8)
    regions[:8] = (255, 0, 0, 255)
    cv2.imwrite(str(tmp_path / "regions.png"), cv2.cvtColor(regions, cv2.COLOR_RGBA2BGRA))
    seen = {}
    import autoforge.auto_forge as af

    real = af.attach_height_guidance

    def spy(optimizer, *a, **k):
        real(optimizer, *a, **k)
        seen["g"] = optimizer.guidance

    monkeypatch.setattr(af, "attach_height_guidance", spy)
    monkeypatch.setattr("sys.argv", [
        "autoforge", "--input_image", str(img_path), "--csv_file", str(csv), "--output_folder", str(tmp_path / "o"),
        "--stl_output_size", "20", "--iterations", "60", "--discrete_check", "10", "--no-visualize",
        "--random_seed", "1", "--minimal_postprocess", "--no-cuda_graph", "--no-perform_pruning",
        "--height_guidance", json.dumps({"pins": [{"filament": "Yellow", "position": 0.9}]}),
        "--height_regions", str(tmp_path / "regions.png"),
    ])
    assert np.isfinite(start(parse_args()))
    g = seen["g"]
    assert g is not None and g.has_pixel and [m for m, _ in g.pins] == [5]


# --------------------------------------------------------------------------
# the stack search sees the pixel targets


def _proxy_setup(device, guided=True):
    from types import SimpleNamespace

    # Left half red, right half blue; the depth target wants the red half
    # at the top of the print and the blue half at the base.
    target = torch.zeros(16, 32, 3, device=device)
    target[:, :16] = torch.tensor([200.0, 30, 30], device=device)
    target[:, 16:] = torch.tensor([30.0, 30, 200], device=device)
    tmap = np.zeros((16, 32), np.float32)
    tmap[:, :16] = 1.0
    colors = torch.tensor([[1.0, 1, 1], [0.8, 0.1, 0.1], [0.1, 0.1, 0.8]], device=device)
    g = HeightGuidance(GuidanceSpec(strength=1), colors, 20, tmap, np.ones_like(tmap), device=device)
    ns = SimpleNamespace(
        target=target, focus_map=None, alpha=None, material_colors=colors,
        material_TDs=torch.full((3,), 0.4, device=device), background=torch.ones(3, device=device),
        h=0.04, device=torch.device(device), base_material=0, guidance=g if guided else None,
    )
    red_high = torch.tensor([2] * 10 + [1] * 10, device=device)  # blue low, red on top
    blue_high = torch.tensor([1] * 10 + [2] * 10, device=device)
    return ns, torch.stack([red_high, blue_high])


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_the_stack_proxy_prefers_colours_at_their_target_heights(device, monkeypatch):
    """The palette proxy every stack search ranks stacks by: without
    guidance both orders show both colours equally well; with depth
    guidance the one that prints red high (where the red pixels want to be)
    must win - else the search never builds depth into the stack and the
    guidance can only force pixels onto wrong colours."""
    from autoforge.Helper import FusedComposite as fc
    from autoforge.Helper.PixelHeightRefine import PaletteProxy

    if device == "cpu":
        monkeypatch.setattr(fc, "fused_available", lambda *a: False)
    ns, stacks = _proxy_setup(device, guided=False)
    plain = PaletteProxy(ns)._eval(stacks)
    assert float(plain[0]) == pytest.approx(float(plain[1]), rel=0.02)
    ns, stacks = _proxy_setup(device, guided=True)
    guided = PaletteProxy(ns)._eval(stacks)
    assert float(guided[0]) < 0.8 * float(guided[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused proxy needs CUDA")
def test_fused_guided_proxy_matches_the_reference():
    from autoforge.Helper.PixelHeightRefine import PaletteProxy

    ns, _ = _proxy_setup("cuda")
    ns.guidance.set_top(14)
    proxy = PaletteProxy(ns)
    stacks = torch.randint(0, 3, (9, 20), device="cuda", generator=torch.Generator("cuda").manual_seed(1))
    fused = proxy._colour_eval(stacks)
    # Directly: (t - p)^2 per point and height, plus the height cost, min over heights.
    pal = proxy.palette_lab(stacks)
    d = ((proxy.tl[None, :, None, :] - pal[:, None, :, :]) ** 2).sum(-1) + proxy._height_cost(pal.shape[1])[None]
    reference = (d.min(dim=2)[0] * proxy.w).sum(-1) / 3.0
    torch.testing.assert_close(fused, reference, rtol=1e-4, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused refine error needs CUDA")
def test_fused_refine_error_includes_the_pixel_cost():
    """The pixel refine's per-pixel error with depth guidance comes out of
    one fused kernel: the same as the colour error plus pixel_cost_map."""
    from autoforge.Helper import FusedComposite as fc
    from autoforge.Helper.HeightGuidance import PIXEL_TOLERANCE

    gen = torch.Generator("cuda").manual_seed(0)
    H, W, L, M = 23, 37, 20, 4
    mc = torch.rand(M, 3, device="cuda", generator=gen)
    td = torch.rand(M, device="cuda", generator=gen) * 5 + 0.5
    bgc = torch.rand(3, device="cuda", generator=gen)
    dg = torch.randint(0, M, (L,), device="cuda", generator=gen)
    z = torch.randint(0, L + 1, (H, W), device="cuda", generator=gen)
    rng = np.random.default_rng(0)
    g = HeightGuidance(GuidanceSpec(strength=3), mc, L, rng.random((H, W)), rng.random((H, W)), device="cuda")
    g.set_top(17)
    p, bg = fc.stack_params(dg, mc, td, bgc, 0.04), fc.background_tensor(bgc)
    tl = torch.rand(H, W, 3, device="cuda", generator=gen) * 50
    w = torch.rand(H, W, device="cuda", generator=gen)
    tgt, pull = g.maps((H, W))
    fused = fc.heights_error(z, p, bg, 0.04, tl, w, 1.0, guided=(tgt, (pull * g.pixel_scale).contiguous(), g._top, PIXEL_TOLERANCE))
    reference = fc.heights_error(z, p, bg, 0.04, tl, w, 1.0) + g.pixel_cost_map(z)
    assert float(g.pixel_cost_map(z).sum()) > 0
    torch.testing.assert_close(fused, reference, rtol=1e-5, atol=1e-2)



def test_a_band_move_shifts_the_band_and_keeps_every_layer():
    from autoforge.Helper.PixelHeightRefine import _move_band

    cur = torch.arange(10)
    up = _move_band(cur, 2, 3, 4)  # layers 2..4 four layers higher
    assert up.tolist() == [0, 1, 5, 6, 7, 8, 2, 3, 4, 9]
    down = _move_band(cur, 6, 2, -5)
    assert down.tolist() == [0, 6, 7, 1, 2, 3, 4, 5, 8, 9]
    # Clipped at both ends, never longer or shorter than the stack.
    assert _move_band(cur, 8, 5, 6).tolist() == cur.tolist()
    assert _move_band(cur, 1, 2, -6).tolist() == [1, 2, 0, 3, 4, 5, 6, 7, 8, 9]


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_refine_candidates_take_the_guided_free_height(device):
    """The pixel refine's free-height candidate: with a colour printed both
    low and high, a pixel pulled to the base gets the low one, even though
    the high one is a little more saturated."""
    from autoforge.Helper.ImageHelper import srgb_to_lab
    from autoforge.Helper.PixelHeightRefine import _free_heights, _pixel_weights

    ns, _ = _proxy_setup(device)
    ns.max_layers = 20
    ns.material_TDs = torch.full((3,), 1.0, device=device)  # translucent: colour builds up with layers
    tmap = np.zeros((16, 32), np.float32)  # everything wanted at the base
    ns.guidance = HeightGuidance(GuidanceSpec(strength=3), ns.material_colors, 20, tmap, np.ones_like(tmap), device=device)
    ns.guidance.set_top(20)
    stack = torch.tensor([1] * 6 + [0] * 4 + [1] * 10, device=device)  # red, white, red again
    tl = srgb_to_lab(ns.target)
    w = _pixel_weights(ns, (16, 32))
    red = _free_heights(ns, stack, tl, w)[:, :16]
    assert int(red.max()) <= 8
    ns.guidance = None
    assert int(_free_heights(ns, stack, tl, w)[:, :16].min()) > 10


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused proxy needs CUDA")
def test_tiled_guided_proxy_matches_one_stack_per_program():
    """Large batches score several stacks per program (the height cost
    shared between them): the same losses as one stack per program."""
    from autoforge.Helper import FusedComposite as fc

    gen = torch.Generator("cuda").manual_seed(2)
    N, n, B = 1000, 30, fc.PROXY_TILE_MIN_BATCH + 3  # a partial last tile
    t = torch.rand(N, 3, device="cuda", generator=gen) * 100
    w = torch.rand(N, device="cuda", generator=gen)
    ht, hw = torch.rand(N, device="cuda", generator=gen), torch.rand(N, device="cuda", generator=gen)
    pal = torch.rand(B, n, 3, device="cuda", generator=gen) * 100
    guided = (ht, hw, torch.tensor(25.0, device="cuda"), 120.0, 0.1)
    tiled = fc.proxy_losses(pal, t, w, guided=guided)
    single = torch.cat([fc.proxy_losses(pal[i : i + 1], t, w, guided=guided) for i in range(B)])
    torch.testing.assert_close(tiled, single, rtol=1e-5, atol=1e-4)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_maps_downsize_to_non_divisible_shape_on_mps():
    # "area" downsizing is adaptive_avg_pool2d, which MPS only implements
    # for divisible sizes: 1000x750 -> 333x250 must still work there.
    rng = np.random.default_rng(0)
    tmap = rng.random((1000, 750)).astype(np.float32)
    weight = rng.random((1000, 750)).astype(np.float32)
    g = HeightGuidance(GuidanceSpec(strength=1), COLORS, 20, tmap, weight, device="mps")
    t, w = g.maps((333, 250))
    assert t.device.type == "mps" and w.device.type == "mps"
    assert tuple(t.shape) == (333, 250) and tuple(w.shape) == (333, 250)
    ref = HeightGuidance(GuidanceSpec(strength=1), COLORS, 20, tmap, weight, device="cpu").maps((333, 250))
    assert torch.allclose(t.cpu(), ref[0], atol=1e-5)
    assert torch.allclose(w.cpu(), ref[1], atol=1e-5)
