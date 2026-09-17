"""Pruning the same result again must not make it worse.

Pruning is a greedy search, so running it repeatedly is a legitimate (and
recommended) way to keep improving a result. That only holds if every step is
non-worsening, and one was not: ``post_remove_spikes`` applied its cleaned
height map unconditionally, and on a detailed image it reliably *raises* the
loss (measured 1015.3209 -> 1016.0118 on the benchmark image, identically on
every pass). Each extra prune therefore paid the same accuracy cost again.

Spike removal is about printability rather than accuracy — single-pixel
towers print badly however good they look in the loss — so the *first* pass
is still allowed to make that trade. Only repeat passes are protected.
"""

import types

import numpy as np
import pytest
import torch

from autoforge.Modules.Optimizer import FilamentOptimizer


class _StubOptimizer:
    """Stands in for FilamentOptimizer around post_remove_spikes.

    The real method needs only a handful of attributes; building a whole
    optimizer (heightmap init, CAdamW state, autocast) would make this a slow
    integration test of everything *except* the accept/revert decision.
    """

    post_remove_spikes = FilamentOptimizer.post_remove_spikes

    def __init__(self, loss_after: float, loss_before: float = 1.0, spikes: int = 7, tmp_path=None):
        self.device = torch.device("cpu")
        self.max_layers = 10
        self.h = 0.04
        self.args = types.SimpleNamespace(
            spike_threshold_layers=1,
            spike_removal_passes=1,
            output_folder=str(tmp_path) if tmp_path else ".",
        )
        self.original = torch.full((4, 4), 0.25)
        self.best_params = {
            "pixel_height_logits": self.original,
            "height_offsets": torch.zeros(self.max_layers, 1),
        }
        self.pixel_height_logits = self.original
        self._losses = [loss_before, loss_after]
        self._spikes = spikes

    # --- the bits post_remove_spikes calls -------------------------------
    def get_discretized_solution(self, best=False):
        return torch.zeros(self.max_layers, dtype=torch.int64), torch.full((4, 4), 5.0)

    def _apply_height_offset(self, pixel_logits, height_offsets):
        return pixel_logits

    def _remove_height_offset(self, pixel_logits, height_offsets):
        return pixel_logits


@pytest.fixture
def patched_spike_helpers(monkeypatch):
    """`remove_height_spikes` returns a *different* height map, and
    `_compute_loss_for_heightmap` returns the queued before/after losses."""
    import autoforge.Helper.PruningHelper as ph

    def fake_remove(height_map, threshold_layers=1, num_passes=4):
        return height_map + 1.0, 7

    monkeypatch.setattr(ph, "remove_height_spikes", fake_remove)

    calls = {"n": 0}

    def fake_loss(optimizer, disc_global, **kwargs):
        value = optimizer._losses[min(calls["n"], len(optimizer._losses) - 1)]
        calls["n"] += 1
        return value

    monkeypatch.setattr(ph, "_compute_loss_for_heightmap", fake_loss)
    return calls


def test_spike_removal_is_kept_when_it_improves(patched_spike_helpers, tmp_path):
    opt = _StubOptimizer(loss_before=1.0, loss_after=0.5, tmp_path=tmp_path)
    assert opt.post_remove_spikes(allow_regression=False) is True
    assert not torch.equal(opt.best_params["pixel_height_logits"], opt.original)


def test_a_repeat_pass_reverts_spike_removal_that_would_hurt(patched_spike_helpers, tmp_path):
    opt = _StubOptimizer(loss_before=1.0, loss_after=1.5, tmp_path=tmp_path)
    assert opt.post_remove_spikes(allow_regression=False) is False
    # Both the best_params copy and the live tensor go back.
    assert torch.equal(opt.best_params["pixel_height_logits"], opt.original)
    assert torch.equal(opt.pixel_height_logits, opt.original)


def test_the_first_pass_may_trade_accuracy_for_printability(patched_spike_helpers, tmp_path):
    # allow_regression=True is the default, and what the CLI uses: a spiky
    # model is a worse *print* even when it scores better.
    opt = _StubOptimizer(loss_before=1.0, loss_after=1.5, tmp_path=tmp_path)
    assert opt.post_remove_spikes() is True
    assert not torch.equal(opt.best_params["pixel_height_logits"], opt.original)


def test_an_equal_loss_counts_as_no_regression(patched_spike_helpers, tmp_path):
    opt = _StubOptimizer(loss_before=1.0, loss_after=1.0, tmp_path=tmp_path)
    assert opt.post_remove_spikes(allow_regression=False) is True


def test_spike_stats_record_whether_it_was_kept(patched_spike_helpers, tmp_path):
    opt = _StubOptimizer(loss_before=1.0, loss_after=1.5, tmp_path=tmp_path)
    opt.post_remove_spikes(allow_regression=False)
    line = (tmp_path / "spike_removal_stats.txt").read_text()
    assert "kept=0" in line
    assert "loss_before=1.000000" in line and "loss_after=1.500000" in line


def test_no_solution_is_a_no_op(monkeypatch, tmp_path):
    opt = _StubOptimizer(loss_after=1.0, tmp_path=tmp_path)
    monkeypatch.setattr(opt, "get_discretized_solution", lambda best=False: (None, None))
    assert opt.post_remove_spikes(allow_regression=False) is False


def test_prune_counts_its_own_runs():
    """``prune()`` is what decides which pass may regress, so it has to know
    how many times this result has been pruned."""
    import inspect

    source = inspect.getsource(FilamentOptimizer.prune)
    assert "_prune_runs" in source
    # First pass allowed to regress, later ones not.
    assert "allow_regression=self._prune_runs <= 1" in source


# ---------------------------------------------------------------------------
# The two optional polish passes that now run before the reduction phases
# ---------------------------------------------------------------------------


class _PolishStub(_StubOptimizer):
    """Adds the pieces polish_height_offsets needs on top of _StubOptimizer."""

    polish_height_offsets = FilamentOptimizer.polish_height_offsets
    _prune_phase_progress = FilamentOptimizer._prune_phase_progress

    def __init__(self, losses, fine_tune_improves=True, tmp_path=None):
        super().__init__(loss_after=0.0, tmp_path=tmp_path)
        self.args.spike_removal = True
        self._queued = list(losses)
        self._fine_tune_improves = fine_tune_improves
        self.preview_callback = None
        self.spike_calls = 0
        self.best_params["height_offsets"] = torch.zeros(self.max_layers, 1)
        self.offsets_before = self.best_params["height_offsets"].clone()
        self.best_discrete_loss = 10.0

    def fine_tune_height_offsets(self, num_steps=50, lr=0.007, progress_callback=None):
        if progress_callback is not None:
            progress_callback(50.0, 1.0)
        if self._fine_tune_improves:
            # Pretend it moved the offsets.
            self.best_params["height_offsets"] = self.best_params["height_offsets"] + 1.0
        return self._fine_tune_improves

    def post_remove_spikes(self, allow_regression=True):
        self.spike_calls += 1
        self.spike_allow_regression = allow_regression
        self.best_params["pixel_height_logits"] = self.best_params["pixel_height_logits"] + 0.5
        return True


@pytest.fixture
def queued_loss(monkeypatch):
    """`_compute_loss_for_heightmap` hands back the stub's queued values."""
    import autoforge.Helper.PruningHelper as ph

    def fake_loss(optimizer, disc_global, **kwargs):
        return optimizer._queued.pop(0) if optimizer._queued else 0.0

    monkeypatch.setattr(ph, "_compute_loss_for_heightmap", fake_loss)


def test_height_polish_cleans_spikes_immediately_after_fine_tuning(queued_loss, tmp_path):
    # Moving the height map can introduce new single-pixel towers, so the
    # spike pass has to run before the result is judged — and it is judged as
    # a pair, which is why the spike pass itself isn't second-guessed.
    opt = _PolishStub([10.0, 8.0], tmp_path=tmp_path)  # pre=10, post=8
    assert opt.polish_height_offsets(num_steps=5) is True
    assert opt.spike_calls == 1
    assert opt.spike_allow_regression is True


def test_height_polish_reverts_both_when_the_pair_is_worse(queued_loss, tmp_path):
    opt = _PolishStub([10.0, 12.0], tmp_path=tmp_path)  # pre=10, post=12
    opt.best_discrete_loss = 10.0
    assert opt.polish_height_offsets(num_steps=5) is False
    # The offsets *and* the height map go back — a fine-tune that scored
    # better on its own can still lose once its new spikes are cleaned up.
    assert torch.equal(opt.best_params["height_offsets"], opt.offsets_before)
    assert torch.equal(opt.best_params["pixel_height_logits"], opt.original)
    # ...and so does the recorded best loss, which fine_tune_height_offsets
    # overwrites when it thinks it improved.
    assert opt.best_discrete_loss == 10.0


def test_height_polish_skips_the_spike_pass_when_fine_tuning_changed_nothing(queued_loss, tmp_path):
    opt = _PolishStub([10.0, 10.0], fine_tune_improves=False, tmp_path=tmp_path)
    assert opt.polish_height_offsets(num_steps=5) is True
    assert opt.spike_calls == 0


def test_height_polish_reports_progress(queued_loss, tmp_path):
    seen = []
    opt = _PolishStub([10.0, 9.0], tmp_path=tmp_path)
    opt.preview_callback = lambda _o, percent, phase=None, loss=None: seen.append((percent, phase, loss))
    opt._current_prune_phase = "Fine-tuning height"
    opt.polish_height_offsets(num_steps=5, progress_callback=opt._prune_phase_progress)
    assert seen == [(50.0, "Fine-tuning height", 1.0)]


def test_progress_reporting_tolerates_older_callbacks(tmp_path):
    """A callback that predates the `loss` (or even `phase`) keyword must
    still work — the CLI's matplotlib preview is one of them."""
    opt = _PolishStub([10.0, 9.0], tmp_path=tmp_path)
    opt._current_prune_phase = "Searching color seeds"

    phase_only = []
    opt.preview_callback = lambda _o, percent, phase=None: phase_only.append((percent, phase))
    opt._prune_phase_progress(25.0, 1.5)
    assert phase_only == [(25.0, "Searching color seeds")]

    bare = []
    opt.preview_callback = lambda _o, percent: bare.append(percent)
    opt._prune_phase_progress(40.0, 1.5)
    assert bare == [40.0]

    # And no callback at all is simply a no-op.
    opt.preview_callback = None
    opt._prune_phase_progress(50.0, 1.0)


def test_prune_exposes_the_polish_passes():
    """Both were hardcoded (200 seeds inside prune, 50 steps in
    export_results); they are now parameters so the UI can switch them off
    and set their limits."""
    import inspect

    signature = inspect.signature(FilamentOptimizer.prune)
    assert signature.parameters["search_seed"].default is True
    assert signature.parameters["seed_search_count"].default == 200
    assert signature.parameters["fine_tune_steps"].default == 50
    # The post-reduction fine-tune stays unconditional, because the CLI has
    # always had it; the *pre*-reduction one is opt-in, because the CLI runs
    # its own (longer) fine-tune before calling prune() and would otherwise
    # do the work twice.
    assert signature.parameters["fine_tune_height"].default is True
    assert signature.parameters["pre_fine_tune_height"].default is False

    source = inspect.getsource(FilamentOptimizer.prune)
    # The seed search is scored against a freshly measured full-resolution
    # loss. It used to be handed the training-time `best_discrete_loss`,
    # measured at the *processing* resolution, which made its "never regress
    # below start_loss" guarantee a comparison between two different image
    # sizes.
    normalised = " ".join(source.split())
    assert "rng_seed_search( current_loss," in normalised
    assert "polish_height_offsets(" in source


# ---------------------------------------------------------------------------
# Per-phase transactions: a phase that wasn't asked to reduce anything must
# improve or hold, measured by the metric the finished result is judged with.
# ---------------------------------------------------------------------------


class _PhaseStub:
    """Just enough optimizer for _run_non_worsening."""

    PHASE_LOSS_EPS = FilamentOptimizer.PHASE_LOSS_EPS
    _run_non_worsening = FilamentOptimizer._run_non_worsening
    solution_snapshot = FilamentOptimizer.solution_snapshot
    restore_solution_snapshot = FilamentOptimizer.restore_solution_snapshot

    def __init__(self, losses):
        self._losses = list(losses)
        self.best_params = {"global_logits": torch.zeros(4, 3), "height_offsets": torch.zeros(4, 1)}
        self.pixel_height_logits = torch.zeros(2, 2)
        self.max_layers = 4
        self.best_seed = 1

    def solution_loss(self):
        return self._losses.pop(0) if self._losses else 0.0

    def mutate(self):
        """What a pruning phase does: change the solution in place."""
        self.best_params = {"global_logits": torch.ones(2, 3), "height_offsets": torch.ones(2, 1)}
        self.pixel_height_logits = torch.ones(2, 2)
        self.max_layers = 2
        self.best_seed = 99


def test_a_phase_that_made_things_worse_is_rolled_back():
    opt = _PhaseStub([10.0, 12.0])  # before=10, after=12
    assert opt._run_non_worsening("Reducing swaps", opt.mutate, forced=False) is False
    # best_params and max_layers go back *together* — prune_redundant_layers
    # replaces the dict and moves max_layers with it, so a half-restore would
    # leave global_logits out of step with the layer count.
    assert opt.max_layers == 4
    assert opt.best_params["global_logits"].shape == (4, 3)
    assert opt.best_seed == 1
    assert torch.equal(opt.pixel_height_logits, torch.zeros(2, 2))


def test_a_phase_that_improved_is_kept():
    opt = _PhaseStub([10.0, 8.0])
    assert opt._run_non_worsening("Reducing colors", opt.mutate, forced=False) is True
    assert opt.max_layers == 2
    assert opt.best_seed == 99


def test_holding_steady_counts_as_kept():
    opt = _PhaseStub([10.0, 10.0])
    assert opt._run_non_worsening("Optimising swap positions", opt.mutate, forced=False) is True
    assert opt.max_layers == 2


def test_a_forced_phase_may_trade_accuracy_for_a_printable_count():
    # Asking for fewer colors than the result has *is* a request to spend
    # accuracy; the guard must not undo it.
    opt = _PhaseStub([10.0, 25.0])
    assert opt._run_non_worsening("Reducing colors", opt.mutate, forced=True) is True
    assert opt.max_layers == 2
    # It doesn't even measure when it can't act on the answer.
    assert opt._losses == [10.0, 25.0]


def test_an_unmeasurable_solution_is_left_alone():
    opt = _PhaseStub([])
    opt.solution_loss = lambda: None
    assert opt._run_non_worsening("Reducing layers", opt.mutate, forced=False) is True
    assert opt.max_layers == 2


def test_each_phase_declares_whether_it_is_forced():
    """The exemption has to be per phase and per limit — "colors are over
    the limit" is no licence for the swap phase to lose accuracy."""
    import inspect

    source = inspect.getsource(FilamentOptimizer.prune)
    assert "forced=_current_counts()[0] > max_colors_allowed" in source
    assert "forced=_current_counts()[1] > max_swaps_allowed" in source
    assert "forced=int(self.max_layers) > max_layers_allowed" in source
    # Moving swap boundaries and searching seeds can never be catching up
    # with a limit, so they are never exempt.
    assert source.count("forced=False") == 2


def test_seed_search_only_accepts_an_improvement():
    import inspect

    source = inspect.getsource(FilamentOptimizer.rng_seed_search)
    # The batched search is re-scored through the exact composite path, and
    # the seed is only adopted when that verified loss beats the start.
    assert "if verified_loss < start_loss:" in source
    assert "if autoset_seed and best_seed is not None and best_loss < start_loss:" in source


def test_fine_tune_only_accepts_an_improvement():
    import inspect

    source = inspect.getsource(FilamentOptimizer.fine_tune_height_offsets)
    assert "if best_loss < pre_loss:" in source
    assert "self.best_params[\"height_offsets\"] = orig_offsets" in source
