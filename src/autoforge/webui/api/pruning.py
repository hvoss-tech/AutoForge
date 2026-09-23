import os
import copy
import threading
import time
import uuid
from fastapi import APIRouter, HTTPException
from ..models import PruningSettings
from ..services.optimization_service import get_optimization_service
from ..helpers.pipeline_runner import friendly_error_message
from ..helpers.sliders import result_counts_from_optimizer, should_repeat_prune
from ..helpers.telemetry import capture_exception
from ..config import config

router = APIRouter()


@router.post("/start")
async def start_pruning(settings: PruningSettings):
    svc = get_optimization_service()

    # Pruning drives the same GPU (and the same FilamentOptimizer as any
    # active optimization) — two such jobs running concurrently interleave
    # mutations of `optimizer.best_params` and write the same output files,
    # so this is a hard mutex, not a recommendation. A double-click on
    # "Start pruning" used to start a second prune thread on the same
    # result.
    busy = svc.get_any_active_job()
    if busy is not None:
        if busy.job_id.startswith("prune-"):
            raise HTTPException(
                409,
                f"Pruning is already {busy.status}. Let it finish or cancel it before starting another pass.",
            )
        raise HTTPException(
            409,
            f"An optimization is {busy.status}. Let it finish or cancel it before pruning.",
        )

    # Pruning must operate on the specific result the frontend is looking
    # at (its `currentJob`) — not "whatever completed job happens to be
    # most recent in history". This backend keeps job records across
    # restarts, so without pinning to an explicit job_id, a user could open
    # the app fresh (no optimization run yet this session) and still
    # successfully "prune" some unrelated leftover job from a previous
    # session/image. Requiring — and validating — job_id closes that gap
    # and also naturally blocks pruning before any optimization has run.
    if not settings.job_id:
        raise HTTPException(400, "Run an optimization first — there's no result to prune yet.")
    target_job = svc.get_job(settings.job_id)
    if not target_job or target_job.status != "completed":
        raise HTTPException(400, "Run an optimization first — there's no completed result to prune yet.")
    job_id = target_job.job_id

    # Create a pruning job via the public API
    import datetime
    from datetime import timezone

    prune_job_id = f"prune-{uuid.uuid4().hex[:8]}"
    # input_image/flatforge carry over so this job is restorable exactly
    # like an optimization job once it completes (see get_latest_job and
    # jobBelongsToImage) — a successful prune clones a real, independent
    # result under this id (below), it is no longer just a progress tracker.
    # input_image/flatforge are Optional[str]/Optional[bool] on JobStatus but
    # plain (non-optional) fields on the settings model create_job() also
    # populates — omit whichever is None rather than pass it through, which
    # that model rejects.
    prune_job_settings: dict = {"iterations": 1}
    if target_job.input_image is not None:
        prune_job_settings["input_image"] = target_job.input_image
    if target_job.flatforge is not None:
        prune_job_settings["flatforge"] = target_job.flatforge
    svc.create_job(prune_job_settings, job_id=prune_job_id)
    cancel_event = svc.cancel_event(prune_job_id)
    pause_event = svc.pause_event(prune_job_id)

    def _run():
        try:
            svc.update_status(prune_job_id, "running")

            # Get the pipeline result for the optimization job
            pipeline_result = svc.get_pipeline_result(job_id)
            if not pipeline_result:
                svc.update_status(prune_job_id, "failed",
                                  error="No optimization pipeline result found. Please run optimization first.")
                return

            from ..helpers.pipeline_runner import export_results

            # Copy args to avoid mutating the stored pipeline result
            args = copy.copy(pipeline_result["args"])
            args.pruning_max_colors = settings.pruning_max_colors
            args.pruning_max_swaps = settings.pruning_max_swaps
            args.pruning_max_layer = settings.pruning_max_layer
            args.perform_pruning = True
            args.prune_seed_search = settings.seed_search
            args.prune_seed_search_count = settings.seed_search_count
            args.prune_fine_tune_height = settings.fine_tune_height
            args.prune_fine_tune_steps = settings.fine_tune_steps

            # Every prune invocation writes into its *own*, fresh directory
            # rather than overwriting the job it started from. Pruning used
            # to always write into checkpoints/<job_id> — the same
            # directory the original optimization (and any earlier prune
            # pass on it) had already written to — so a second pruning pass
            # silently replaced the first pass's output files in place.
            # Anything that still referenced the *original* job id (a
            # history step captured right after the first pass, an F5
            # reload) then showed whatever pruning had most recently done
            # to that shared directory instead of what was actually true
            # when it was captured. Each pass gets its own id and directory
            # instead, and the frontend switches to treating this new id as
            # the current job once it completes (see appStore.startPruning)
            # — so a subsequent pass targets *this* directory, never an
            # earlier one.
            output_dir = os.path.join(config.checkpoints_path, prune_job_id)
            args.output_folder = output_dir
            os.makedirs(output_dir, exist_ok=True)

            pipeline_result = dict(pipeline_result)
            pipeline_result["args"] = args

            # Report pruning progress through the optimizer's preview callback
            # so the frontend can show it in the top progress bar. Pruning
            # runs through several genuinely distinct phases in a fixed
            # order (see FilamentOptimizer.prune's `_current_prune_phase`),
            # each raising its own preview_callback(..., phase=<name>) calls
            # with a stage-relative, non-monotonic percent (negative,
            # climbing to 0 for the reduction phases; ~90-99 for swap
            # position optimisation). Give each phase an equal-width slice
            # of the 0-100 bar instead of the old percent-magnitude
            # heuristics, which happened to lump colour/swap/layer
            # reduction into one shared 50%-wide bucket (they all report
            # percent <= 0) while swap-position optimisation alone got the
            # other half.
            optimizer = pipeline_result["optimizer"]
            # In the order prune() runs them. The two optional polish phases
            # only take a slice of the bar when they're actually enabled, so
            # the progress doesn't stall at 0% for a phase that never runs.
            # They report a plain 0-100 fraction of their own work
            # (_prune_phase_progress), unlike the older phases below.
            DIRECT_PERCENT_PHASES = {"Searching color seeds", "Fine-tuning height"}
            phases = [
                *(["Searching color seeds"] if settings.seed_search else []),
                *(["Fine-tuning height"] if settings.fine_tune_height else []),
                "Reducing colors",
                "Reducing swaps",
                "Reducing layers",
                "Optimising swap positions",
                # prune()'s own unconditional closing polish — a second
                # height fine-tune (reported under the same "Fine-tuning
                # height" name; the `.index()` lookup below finds its
                # *first* occurrence, which is fine — its progress is
                # cosmetic this late) followed by spike cleanup, gated on
                # `args.spike_removal` exactly as Optimizer.prune() gates it.
                # Without an entry here, "Removing spikes" was an unknown
                # phase name and fell into the *first* bucket instead of the
                # last, so the bar visibly jumped backward right as pruning
                # was finishing.
                *(["Removing spikes"] if getattr(args, "spike_removal", True) else []),
            ]
            phase_slice = 100.0 / len(phases)
            _last = 0.0
            _phase_min_seen: dict[str, float] = {}
            # Re-deriving the discrete solution means a full discretize pass
            # plus a device->host copy; pruning fires its callback far more
            # often than a person can read a number, so sample it at most
            # once a second and reuse the last counts in between.
            _COUNTS_INTERVAL_S = 1.0
            _counts_state = {"at": 0.0, "value": None}

            def _live_counts() -> dict:
                now = time.monotonic()
                if _counts_state["value"] is None or now - _counts_state["at"] >= _COUNTS_INTERVAL_S:
                    try:
                        counts = result_counts_from_optimizer(optimizer)
                    except Exception:
                        counts = None
                    _counts_state["at"] = now
                    if counts is not None:
                        _counts_state["value"] = counts
                value = _counts_state["value"]
                if not value:
                    return {}
                return {
                    "result_colors": value["colors"],
                    "result_swaps": value["swaps"],
                    "result_layers": value["layers"],
                }

            # Pass bookkeeping for auto-repeat. Each pass is a full prune, so
            # the 0-100 bar is per-pass; the pass number rides along in the
            # status so the UI can say which one is running.
            _pass = 1

            def _pass_fields() -> dict:
                if not settings.auto_repeat:
                    return {}
                return {"pruning_pass": _pass, "pruning_max_passes": settings.max_passes}

            def _prune_progress(_optimizer, _percent, phase=None, loss=None):
                nonlocal _last
                phase_name = phase or phases[0]
                # A phase name not in the list (the optimizer reports one
                # this webui version doesn't know about yet — e.g. a future
                # phase added to Optimizer.prune's ordering) is far more
                # likely to be a *late* one than the very first: pruning's
                # named phases run in a fixed, append-only order, so bucket
                # it at the end rather than jumping the bar back to 0%.
                phase_idx = phases.index(phase_name) if phase_name in phases else len(phases) - 1
                bucket_start = phase_idx * phase_slice

                if phase_name in DIRECT_PERCENT_PHASES:
                    # Reported as a plain 0-100 fraction of this phase's work.
                    within = min(1.0, max(0.0, float(_percent) / 100.0))
                elif phase_name == "Optimising swap positions":
                    # Raw percent here is "90 + pass number" (capped at 99) —
                    # a small incrementing pass counter, not a 0-100 fraction
                    # of this phase's own work. Rescale it onto this phase's
                    # slice instead of taking it as an absolute percentage.
                    within = min(1.0, max(0.0, (float(_percent) - 90.0) / 9.0))
                else:
                    # Colour/swap/layer reduction: percent starts very
                    # negative and climbs toward 0 as the search converges.
                    # Track each phase's own low-water mark separately so
                    # they don't share (and corrupt) one running minimum.
                    seen = min(_phase_min_seen.get(phase_name, float(_percent)), float(_percent))
                    _phase_min_seen[phase_name] = seen
                    denom = 0.0 - seen
                    within = (float(_percent) - seen) / denom if denom > 1e-9 else 0.0
                    within = min(1.0, max(0.0, within))

                p = bucket_start + within * phase_slice
                _last = max(_last, min(p, 100.0))
                # The polish phases carry their own best-so-far loss; the
                # reduction phases leave it alone (their progress shows in
                # the counts instead).
                loss_field = {} if loss is None else {"loss": float(loss)}
                svc.update_status(
                    prune_job_id, "running", progress=_last, phase=phase_name,
                    **loss_field, **_live_counts(), **_pass_fields(),
                )

            optimizer.preview_callback = _prune_progress

            # Pruning regenerates this job's real outputs; an earlier slider
            # edit rendered against the unpruned solution would otherwise
            # keep hiding the pruned mesh in the 3D view.
            from .outputs import discard_slider_edits
            discard_slider_edits(job_id)

            def _broadcast_result() -> None:
                """Push the pruned slider stack + image to the frontend, so
                the color layers and both previews follow each pass."""
                try:
                    from ..helpers.sliders import derive_sliders_from_result
                    from .ws import broadcast_preview

                    slider_data = derive_sliders_from_result(pipeline_result)
                    if not (slider_data and slider_data["sliders"]):
                        return
                    final_image = optimizer.get_best_discretized_image()
                    if final_image is None:
                        return
                    import base64
                    import cv2
                    import numpy as np

                    img_np = final_image.cpu().numpy().astype(np.uint8)
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    _ok, buf = cv2.imencode('.png', img_bgr)
                    broadcast_preview(
                        base64.b64encode(buf.tobytes()).decode('utf-8'),
                        # A client matches broadcasts against
                        # currentJob.job_id, which is still `job_id` (the id
                        # this prune action started from) for the whole
                        # duration of this call — the frontend only switches
                        # currentJob to the freshly-cloned `prune_job_id`
                        # once this action fully completes (see
                        # appStore.startPruning), so mid-run broadcasts must
                        # keep targeting the id it's showing right now.
                        job_id=job_id,
                        sliders=slider_data["sliders"],
                        min_layer=slider_data["min_layer"],
                        max_layer=slider_data["max_layer"],
                    )
                except Exception:
                    import traceback
                    traceback.print_exc()

            def _discrete_loss() -> float | None:
                """The loss of the solution as it now stands — the measure
                auto-repeat decides on. Deliberately the same
                `_compute_loss_for_heightmap` the pruning phases score
                against, so "no improvement" means the same thing here as it
                does inside them."""
                try:
                    from autoforge.Helper.PruningHelper import _compute_loss_for_heightmap

                    disc_global, _ = optimizer.get_discretized_solution(best=True)
                    if disc_global is None:
                        return None
                    return float(_compute_loss_for_heightmap(optimizer, disc_global))
                except Exception:
                    return None

            best_loss: float | None = None
            cancelled_midway = False
            max_passes = settings.max_passes if settings.auto_repeat else 1

            # Where we started, so the dialog can show what pruning gained
            # rather than a bare number with nothing to compare it to.
            start_loss = _discrete_loss()
            svc.update_status(
                prune_job_id, "running", pruning_start_loss=start_loss, loss=start_loss,
            )

            # Spike removal is a printability trade-off (see
            # Optimizer.post_remove_spikes), not part of the color/swap/layer
            # search — running it every auto-repeat pass paid its accuracy
            # cost, and the extra compute, on intermediate results that the
            # next pass was about to prune further anyway. It only has to run
            # once the loop has actually settled, so every pass here applies
            # it only when this is known to be the *last* one: either
            # auto-repeat is off (a single pass is always "the end"), or the
            # convergence check just below decided to stop and scheduled one
            # extra pass purely to finish with spikes cleaned up.
            final_spike_pass = False
            while True:
                # Each pass starts its own 0-100 sweep.
                _last = 0.0
                _phase_min_seen.clear()
                _counts_state["value"] = None
                # Publish the starting counts immediately: the dialog opens on
                # "Reducing colors" with the bar at 0, and showing "—" there
                # until the first callback landed looked like a stall.
                svc.update_status(
                    prune_job_id, "running", progress=0.0, phase=phases[0],
                    **_live_counts(), **_pass_fields(),
                )

                apply_spikes = (not settings.auto_repeat) or final_spike_pass
                outputs = export_results(
                    pipeline_result,
                    cancel_event=cancel_event,
                    pause_event=pause_event,
                    apply_spike_removal=apply_spikes,
                )
                # Cancelled between phases — the solution as of the last
                # completed phase was still exported, so the partial result
                # is real and usable, just not fully pruned. The user asked
                # to stop, so their sliders aren't force-overwritten either.
                if not outputs.get("pruning_completed", True):
                    cancelled_midway = True
                    break

                _broadcast_result()
                loss = _discrete_loss()
                _counts_state["value"] = None
                svc.update_status(
                    prune_job_id, "running", progress=100.0, phase="Pruned",
                    loss=loss, **_live_counts(), **_pass_fields(),
                )

                if not settings.auto_repeat or final_spike_pass:
                    break
                if cancel_event is not None and cancel_event.is_set():
                    cancelled_midway = True
                    break
                repeat, reason = should_repeat_prune(_pass, max_passes, loss, best_loss)
                if loss is not None and (best_loss is None or loss < best_loss):
                    best_loss = loss
                if not repeat:
                    print(f"Auto-repeat pruning: stopping after pass {_pass} — {reason}.")
                    if getattr(args, "spike_removal", True):
                        # One more pass, purely to clean up spikes on the
                        # converged result — everything else is already
                        # within its limits so the reduction phases exit
                        # immediately, leaving only the spike pass to do
                        # real work.
                        final_spike_pass = True
                        _pass += 1
                        continue
                    break
                _pass += 1

            _counts_state["value"] = None  # force a fresh read of the final solution
            if cancelled_midway:
                svc.update_status(
                    prune_job_id, "cancelled", phase=None, **_live_counts(), **_pass_fields(),
                )
            else:
                # This job now has its own real output directory (written
                # above) and needs its own pipeline result to match — a
                # follow-up prune, or a slider-derivation lookup, targeting
                # this id must find the (still-mutating) optimizer, not
                # come up empty. Aliasing, not re-registering: see
                # alias_pipeline_result for why this must not go through
                # set_pipeline_result's "release every older result" path.
                svc.alias_pipeline_result(prune_job_id, job_id)
                svc.update_status(
                    prune_job_id, "completed", progress=100.0, phase=None,
                    **_live_counts(), **_pass_fields(),
                )
        except Exception as e:
            import traceback
            traceback.print_exc()
            capture_exception(e, {"phase": "pruning", "job_id": prune_job_id})
            svc.update_status(prune_job_id, "failed", error=friendly_error_message(e))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return {"job_id": prune_job_id, "status": "running"}
