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

    filament_dicts = [
        {"color": f.color, "td": f.td, "name": f"{f.brand} - {f.name}",
         "brand": f.brand, "short_name": f.name, "owned": f.owned,
         "uuid": f.uuid, "filament_type": f.filament_type}
        for f in active
    ]

    # One optimization at a time: a second run (e.g. "Start Optimization" in
    # the Settings dialog while the first was paused) competed for the same
    # GPU memory and left two jobs broadcasting into the same preview.
    busy = svc.get_active_optimization_job()
    if busy is not None:
        raise HTTPException(
            409,
            f"An optimization is already {busy.status}. Resume or cancel it before starting a new one.",
        )

    job = svc.create_job(settings_dict)

    def _run():
        nonlocal input_image_path
        try:
            svc.update_status(job.job_id, "running",
                              total_iterations=settings.iterations)


            output_dir = os.path.join(config.checkpoints_path, job.job_id)
            os.makedirs(output_dir, exist_ok=True)

            # Resolve path: try as-is, then under uploads directory
            if not os.path.exists(input_image_path):
                abs_path = os.path.join(config.uploads_path, input_image_path)
                if os.path.exists(abs_path):
                    input_image_path = abs_path
                else:
                    err = f"Input image not found: {input_image_path} (tried {abs_path} too)"
                    logger.error(err)
                    svc.update_status(job.job_id, "failed", error=err)
                    return

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
                settings=settings_dict,
                preview_callback=_preview_callback,
                progress_callback=_progress_callback,
                cancel_event=cancel_event,
                pause_event=pause_event,
            )

            cancelled = result.get("cancelled", False)
            if cancelled:
                svc.update_status(job.job_id, "cancelled")
                return

            # Store pipeline result for pruning later
            svc.set_pipeline_result(job.job_id, result)
            logger.info("Optimization completed: job_id=%s", job.job_id)

            # Generate output files (STL, colored PLY, preview PNG, swap instructions)
            try:
                from ..helpers.pipeline_runner import export_results
                export_results(result)
                logger.info("Output files written to %s", output_dir)
            except Exception as exc:
                logger.warning("Could not export all output files: %s", exc)

            svc.update_status(job.job_id, "completed",
                              progress=100.0,
                              iteration=settings.iterations,
                              total_iterations=settings.iterations)

        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error("Optimization failed: %s", e)
            svc.update_status(job.job_id, "failed", error=friendly_error_message(e))

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


@router.get("/status/{job_id}")
async def get_status(job_id: str):
    svc = get_optimization_service()
    job = svc.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job.model_dump()


@router.get("/latest")
async def get_latest_job():
    svc = get_optimization_service()
    job = svc.get_latest_job()
    if not job:
        raise HTTPException(404, "No jobs yet")
    return job.model_dump()


@router.get("/history")
async def get_history():
    svc = get_optimization_service()
    results = svc.get_history()
    return [r.model_dump() for r in results]


@router.get("/result/{job_id}")
async def get_result(job_id: str):
    svc = get_optimization_service()
    result = svc.get_result(job_id)
    if not result:
        raise HTTPException(404, "Result not found")
    return result.model_dump()
