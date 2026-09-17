import os
import copy
import threading
import uuid
from fastapi import APIRouter, HTTPException
from ..models import PruningSettings
from ..services.optimization_service import get_optimization_service
from ..helpers.pipeline_runner import friendly_error_message
from ..config import config

router = APIRouter()


@router.post("/start")
async def start_pruning(settings: PruningSettings):
    svc = get_optimization_service()

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
    svc.create_job({"iterations": 1}, job_id=prune_job_id)
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

            output_dir = os.path.join(config.checkpoints_path, job_id)
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
            phases = ["Reducing colors", "Reducing swaps", "Reducing layers", "Optimising swap positions", "Fine-tuning height"]
            phase_slice = 100.0 / len(phases)
            _last = 0.0
            _phase_min_seen: dict[str, float] = {}

            def _prune_progress(_optimizer, _percent, phase=None):
                nonlocal _last
                phase_name = phase or phases[0]
                phase_idx = phases.index(phase_name) if phase_name in phases else 0
                bucket_start = phase_idx * phase_slice

                if phase_name == "Optimising swap positions":
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
                svc.update_status(prune_job_id, "running", progress=_last, phase=phase_name)

            optimizer.preview_callback = _prune_progress

            # Pruning regenerates this job's real outputs; an earlier slider
            # edit rendered against the unpruned solution would otherwise
            # keep hiding the pruned mesh in the 3D view.
            from .outputs import discard_slider_edits
            discard_slider_edits(job_id)

            outputs = export_results(
                pipeline_result,
                cancel_event=cancel_event,
                pause_event=pause_event,
            )

            # Push the pruned slider stack to the frontend so the color core
            # and sliders reflect the final (reduced) solution. Skipped when
            # cancelled — the user explicitly asked to stop, so their
            # current sliders shouldn't be force-overwritten by whatever
            # partial state pruning happened to reach.
            if outputs.get("pruning_completed", True):
                try:
                    from ..helpers.sliders import derive_sliders_from_result
                    from .ws import broadcast_preview

                    slider_data = derive_sliders_from_result(pipeline_result)
                    if slider_data and slider_data["sliders"]:
                        final_image = optimizer.get_best_discretized_image()
                        if final_image is not None:
                            import base64
                            import cv2
                            import numpy as np

                            img_np = final_image.cpu().numpy().astype(np.uint8)
                            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                            _ok, buf = cv2.imencode('.png', img_bgr)
                            b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
                            broadcast_preview(
                                b64,
                                # The pruned PLY/preview were regenerated in
                                # place for the *original* optimization job
                                # (`job_id`), not the pruning job
                                # (`prune_job_id`) — a client matches
                                # broadcasts against currentJob.job_id, which
                                # stays the optimization job throughout.
                                job_id=job_id,
                                sliders=slider_data["sliders"],
                                min_layer=slider_data["min_layer"],
                                max_layer=slider_data["max_layer"],
                            )
                except Exception:
                    import traceback
                    traceback.print_exc()

            if outputs.get("pruning_completed", True):
                svc.update_status(prune_job_id, "completed", progress=100.0, phase=None)
            else:
                # Cancelled between phases — the solution as of the last
                # completed phase was still exported above, so the partial
                # result is real and usable, just not fully pruned.
                svc.update_status(prune_job_id, "cancelled", phase=None)
        except Exception as e:
            import traceback
            traceback.print_exc()
            svc.update_status(prune_job_id, "failed", error=friendly_error_message(e))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return {"job_id": prune_job_id, "status": "running"}
