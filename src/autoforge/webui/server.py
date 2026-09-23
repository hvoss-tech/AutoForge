import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import config
from .helpers.telemetry import capture_exception, flush as flush_telemetry

FRONTEND_DIST = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "..",
        "webui",
        "frontend",
        "dist",
    )
)


class SPAStaticFiles(StaticFiles):
    """Serve the SPA. The index.html entry must never be cached for long, or
    browsers keep running an old JS bundle after a rebuild; hashed assets
    under /assets/ get long-lived caching instead."""

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        request_path = scope.get("path", "") or ""
        if request_path in ("", "/", "/index.html"):
            response.headers["Cache-Control"] = "no-cache"
        else:
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(config.uploads_path, exist_ok=True)
    os.makedirs(config.checkpoints_path, exist_ok=True)

    # Auto-seed the filament library from the default CSV if empty
    from .services.filament_service import get_filament_service
    fs = get_filament_service()
    if not fs.list():
        default_csv = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "default_materials.csv",
        )
        if os.path.exists(default_csv):
            with open(default_csv) as f:
                csv_content = f.read()
            imported = fs.import_csv(csv_content, _mark_user_import=False)
            if imported:
                logging.getLogger(__name__).info(
                    "Seeded %d filaments from %s", len(imported), default_csv
                )

    yield
    flush_telemetry()


def create_app() -> FastAPI:
    os.makedirs(config.uploads_path, exist_ok=True)
    os.makedirs(config.checkpoints_path, exist_ok=True)

    app = FastAPI(title="AutoForge WebUI", version="1.0.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception):
        # Background job threads (optimization/pruning) catch their own
        # exceptions and report separately — this only sees errors raised
        # directly out of a request handler.
        logging.getLogger(__name__).exception("Unhandled error in %s", request.url.path)
        capture_exception(exc, {"path": request.url.path})
        return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})

    # API routes (imported here to avoid circular imports during module setup)
    from .api.filaments import router as filaments_router
    from .api.images import router as images_router
    from .api.init import router as init_router
    from .api.optimization import router as optimization_router
    from .api.pruning import router as pruning_router
    from .api.outputs import router as outputs_router
    from .api.settings import router as settings_router
    from .api.project import router as project_router
    from .api.state import router as state_router
    from .api.ws import router as ws_router
    from .api.sliders import router as sliders_router
    from .api.preview import router as preview_router
    from .api.system import router as system_router

    app.include_router(filaments_router, prefix="/api/filaments", tags=["filaments"])
    app.include_router(images_router, prefix="/api/images", tags=["images"])
    app.include_router(init_router, prefix="/api/init", tags=["init"])
    app.include_router(optimization_router, prefix="/api/optimize", tags=["optimization"])
    app.include_router(pruning_router, prefix="/api/pruning", tags=["pruning"])
    app.include_router(outputs_router, prefix="/api/outputs", tags=["outputs"])
    app.include_router(settings_router, prefix="/api/settings", tags=["settings"])
    app.include_router(project_router, prefix="/api/project", tags=["project"])
    app.include_router(state_router, prefix="/api/state", tags=["state"])
    app.include_router(ws_router, prefix="", tags=["websocket"])
    app.include_router(sliders_router, prefix="/api/sliders", tags=["sliders"])
    app.include_router(preview_router, prefix="/api/preview", tags=["preview"])
    app.include_router(system_router, prefix="/api/system", tags=["system"])

    app.mount("/uploads", StaticFiles(directory=config.uploads_path), name="uploads")

    # Serve the built React frontend (SPA). Registered last so the API routes
    # and /uploads mount take precedence.
    if os.path.isdir(FRONTEND_DIST):
        app.mount("/", SPAStaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
    else:
        logging.getLogger(__name__).warning(
            "Frontend build not found at %s. Run `./run_webui.sh` to build it.",
            FRONTEND_DIST,
        )

    return app


app = create_app()
