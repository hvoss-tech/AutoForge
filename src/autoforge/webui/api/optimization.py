import os
import logging
import numpy as np
import base64
import threading
from fastapi import APIRouter, HTTPException
from ..models import OptimizationSettings, StateSnapshot
from ..services.optimization_service import get_optimization_service
from ..services.project_service import get_project_service
from ..services.filament_service import get_filament_service
from ..helpers.pipeline_runner import run_pipeline as _run_pipeline, friendly_error_message
from ..helpers.gpu_memory import empty_device_cache, release_pipeline_result
from ..helpers.telemetry import capture_exception
from ..services.image_service import get_image_service
from ..config import config
from .ws import broadcast_preview

logger = logging.getLogger(__name__)
router = APIRouter()


def _derive_sliders(optimizer, filament_dicts):
    """Best-effort slider derivation; returns None when not possible."""
    try:
        import numpy as np
        from ..helpers.sliders import derive_sliders_from_optimizer
        material_tds = np.array(
            [float(f.get("td", 0.0)) for f in filament_dicts], dtype=np.float64
        )
        material_uuids = [str(f.get("uuid", "")) for f in filament_dicts]
        layer_height = float(getattr(optimizer, "h", 0.04))
        return derive_sliders_from_optimizer(
            optimizer, material_tds, material_uuids, layer_height
        )
    except Exception:
        import traceback
        traceback.print_exc()
        return None


@router.post("/start")
async def start_optimization(settings: OptimizationSettings):
    svc = get_optimization_service()
    filament_svc = get_filament_service()

    # Log what we received
    settings_dict = settings.model_dump()
    logger.info("start_optimization called: input_image=%r, iterations=%d, filaments_active=%d",
                settings_dict.get("input_image", ""),
                settings.iterations,
                len(filament_svc.get_active()))

    # Validate up front, synchronously, before creating/starting a job — the
    # job used to be created and flipped to "running" first, with these same
    # checks only run *inside* the background thread a moment later. That
    # meant "no active filaments" or "no input image" briefly looked like a
    # real run (status: running) before immediately failing, instead of
    # being rejected outright.
    active = filament_svc.get_active()
    if not active:
        raise HTTPException(
            400,
            "Add at least one active filament before running optimization. "
            "Open the Filament Library panel, find a filament and drag it "
            "onto the 'Active Filaments' area, or click the '+' button to "
            "add it to the active list.",
        )
    input_image_path = settings_dict.get("input_image", "")
    if not input_image_path:
        raise HTTPException(400, "Upload an input image before running optimization.")
    base_error = settings.base_height_error()
    if base_error:
        raise HTTPException(400, base_error)

    filament_dicts = [
        {"color": f.color, "td": f.td, "name": f"{f.brand} - {f.name}",
         "brand": f.brand, "short_name": f.name, "owned": f.owned,
         "uuid": f.uuid, "filament_type": f.filament_type}
        for f in active
    ]

    # One GPU job at a time: a second run (e.g. "Start Optimization" in the
    # Settings dialog while the first was paused) competed for the same GPU
    # memory and left two jobs broadcasting into the same preview. Pruning
    # counts too — it runs the same optimizer the new run would clear out.
    busy = svc.get_any_active_job()
    if busy is not None:
        if busy.job_id.startswith("prune-"):
            raise HTTPException(
                409,
                f"Pruning is {busy.status}. Let it finish or cancel it before starting a new optimization.",
            )
        raise HTTPException(
            409,
            f"An optimization is already {busy.status}. Resume or cancel it before starting a new one.",
        )

    from .init import is_init_building

    if is_init_building():
        raise HTTPException(
            409,
            "The preview of the image is still being prepared. Wait for it to finish, then try again.",
        )

    job = svc.create_job(settings_dict)

    def _run():
        nonlocal input_image_path
        run_device = None
        try:
            svc.update_status(job.job_id, "running",
                              total_iterations=settings.iterations,
                              phase="Preparing")

            # A run that has already been cancelled must not touch any other
            # job's retained result: releasing them here was what made a
            # cancelled (or OOM-during-init) run leave the previous, still
            # completed, job un-prunable and un-re-colorable.
            cancel_event = svc.cancel_event(job.job_id)
            if cancel_event is not None and cancel_event.is_set():
                return

            # Hand back every earlier run's device memory *before* this one
            # allocates. Each retained pipeline result pins a whole optimizer
            # (parameters, Adam state, target images) on the GPU, so running
            # image after image without this climbed until a later run OOM'd.
            # Only the newest result is reachable from the UI anyway — the
            # Pruning dialog always targets the job currently on screen.
            svc.clear_all_pipeline_results()


            output_dir = os.path.join(config.checkpoints_path, job.job_id)
            os.makedirs(output_dir, exist_ok=True)

            # Resolve to a real path under the uploads directory only — an
            # arbitrary existing server path (e.g. "../../etc/passwd" or an
            # absolute path outside uploads/) used to be accepted verbatim
            # whenever os.path.exists() happened to be true for it.
            resolved_path = get_image_service().get_path(input_image_path)
            if resolved_path is None:
                err = f"Input image not found: {input_image_path}"
                logger.error(err)
                svc.update_status(job.job_id, "failed", error=err)
                return
            input_image_path = resolved_path

            # The focus-area mask painted in the image panel is uploaded like
            # an image and referenced by its uploads/ file name. Resolve it the
            # same guarded way: only files under uploads/ are accepted.
            run_settings = dict(settings_dict)
            mask_name = run_settings.get("priority_mask") or ""
            if mask_name:
                mask_path = get_image_service().get_path(mask_name)
                if mask_path is None:
                    err = (f"Focus-area mask not found: {mask_name}. "
                           "Clear the focus areas in the image panel, or paint them again.")
                    logger.error(err)
                    svc.update_status(job.job_id, "failed", error=err)
                    return
                run_settings["priority_mask"] = mask_path

            logger.info("Optimization starting: image=%s, filaments=%d, iters=%d",
                        input_image_path, len(filament_dicts), settings.iterations)

            cancel_event = svc.cancel_event(job.job_id)
            pause_event = svc.pause_event(job.job_id)

            def _progress_callback(opt, step):
                """Lightweight progress update — always succeeds regardless of image generation."""
                try:
                    total = max(settings.iterations, 1)
                    loss_val = (
                        opt.best_discrete_loss.item()
                        if hasattr(opt.best_discrete_loss, 'item')
                        else opt.best_discrete_loss
                    ) if opt.best_discrete_loss is not None else None
                    svc.update_status(
                        job.job_id, "running",
                        progress=min(100.0, (step / total) * 100),
                        iteration=step,
                        loss=loss_val,
                        total_iterations=settings.iterations,
                        phase="Optimizing",
                    )
                except Exception:
                    import traceback
                    traceback.print_exc()

            def _preview_callback(opt, step):
                """Progress + preview image. Always sends progress first, then image if available."""
                _progress_callback(opt, step)
                try:
                    img = opt.get_best_discretized_image()
                    if img is not None:
                        import cv2, numpy as np
                        loss_val = (
                            opt.best_discrete_loss.item()
                            if hasattr(opt.best_discrete_loss, 'item')
                            else opt.best_discrete_loss
                        ) if opt.best_discrete_loss is not None else None
                        img_np = img.cpu().numpy().astype(np.uint8)
                        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                        success, buf = cv2.imencode('.png', img_bgr)
                        b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
                        svc.update_status(
                            job.job_id, "running",
                            progress=min(100.0, (step / max(settings.iterations, 1)) * 100),
                            iteration=step, loss=loss_val,
                            preview_image=b64,
                            total_iterations=settings.iterations,
                        )
                        slider_data = _derive_sliders(opt, filament_dicts)
                        broadcast_preview(
                            b64,
                            job_id=job.job_id,
                            sliders=slider_data["sliders"] if slider_data else None,
                            iteration=step,
                            loss=loss_val,
                            min_layer=slider_data["min_layer"] if slider_data else None,
                            max_layer=slider_data["max_layer"] if slider_data else None,
                        )
                except Exception:
                    import traceback
                    traceback.print_exc()

            result = _run_pipeline(
                input_image_path=input_image_path,
                active_filaments=filament_dicts,
                output_dir=output_dir,
                settings=run_settings,
                preview_callback=_preview_callback,
                progress_callback=_progress_callback,
                cancel_event=cancel_event,
                pause_event=pause_event,
            )

            run_device = result.get("device")
            cancelled = result.get("cancelled", False)
            if cancelled:
                svc.update_status(job.job_id, "cancelled")
                # Nothing keeps a cancelled run's state — and nothing will
                # ever ask for it again, so its device memory goes back now
                # rather than whenever this frame happens to be collected.
                release_pipeline_result(result)
                return

            # Store pipeline result for pruning later
            svc.set_pipeline_result(job.job_id, result)
            logger.info("Optimization completed: job_id=%s", job.job_id)

            # Generate output files (STL, colored PLY, preview PNG, swap instructions)
            svc.update_status(job.job_id, "running", phase="Exporting results")
            try:
                from ..helpers.pipeline_runner import export_results
                export_results(result)
                logger.info("Output files written to %s", output_dir)
            except Exception as exc:
                # A run whose export failed (e.g. OOM building the full-res
                # composite, or a disk write error) has no STL/PLY/project
                # files on disk, so downloads would 404. It used to still be
                # marked "completed" here — a green run with a broken
                # "Download" button and no error visible anywhere in the UI.
                logger.error("Export failed for job %s: %s", job.job_id, exc)
                capture_exception(exc, {"phase": "export", "job_id": job.job_id})
                svc.update_status(job.job_id, "failed", error=friendly_error_message(exc))
                return

            svc.update_status(job.job_id, "completed",
                              phase=None,
                              progress=100.0,
                              iteration=settings.iterations,
                              total_iterations=settings.iterations)

        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error("Optimization failed: %s", e)
            capture_exception(e, {"phase": "optimization", "job_id": job.job_id})
            svc.update_status(job.job_id, "failed", error=friendly_error_message(e))
        finally:
            # Training and export both leave large freed blocks in the
            # caching allocator. Returning them keeps `nvidia-smi` honest and,
            # more importantly, leaves room for the next image's init.
            empty_device_cache(run_device)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return {
        "job_id": job.job_id,
        "total_iterations": settings.iterations,
        "status": "running",
    }


@router.post("/pause/{job_id}")
async def pause_optimization(job_id: str):
    svc = get_optimization_service()
    if not svc.pause(job_id):
        raise HTTPException(404, "Job not found")
    # The job's real status, not an assumed one — a job that already
    # finished is left untouched, and the client needs to know that.
    return {"status": svc.get_job(job_id).status}


@router.post("/resume/{job_id}")
async def resume_optimization(job_id: str):
    svc = get_optimization_service()
    if not svc.resume(job_id):
        raise HTTPException(404, "Job not found")
    # The job's real status, not an assumed one — a job that already
    # finished is left untouched, and the client needs to know that.
    return {"status": svc.get_job(job_id).status}


@router.post("/cancel/{job_id}")
async def cancel_optimization(job_id: str):
    svc = get_optimization_service()
    if not svc.cancel(job_id):
        raise HTTPException(404, "Job not found")
    # The job's real status, not an assumed one — a job that already
    # finished is left untouched, and the client needs to know that.
    return {"status": svc.get_job(job_id).status}


# The preview image is a full base64 PNG that is only meaningful on the live
# /ws/preview channel. Excluding it from every status payload keeps the
# 0.5 s WS polls and 1 s status polls small, and stops it leaking into
# history.json (which it grew without bound when it was serialized there).
_STATUS_EXCLUDES = {"preview_image"}


def _job_payload(job) -> dict:
    return job.model_dump(exclude=_STATUS_EXCLUDES)


@router.get("/status/{job_id}")
async def get_status(job_id: str):
    svc = get_optimization_service()
    job = svc.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return _job_payload(job)


@router.get("/latest")
async def get_latest_job():
    svc = get_optimization_service()
    job = svc.get_latest_job()
    if not job:
        raise HTTPException(404, "No jobs yet")
    return _job_payload(job)


@router.get("/history")
async def get_history():
    svc = get_optimization_service()
    results = svc.get_history()
    return [_job_payload(r) for r in results]


@router.get("/result/{job_id}")
async def get_result(job_id: str):
    svc = get_optimization_service()
    result = svc.get_result(job_id)
    if not result:
        raise HTTPException(404, "Result not found")
    return _job_payload(result)
