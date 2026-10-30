"""Çalıştır: python -m uvicorn app.main:app --reload"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import torch

from app.model import load_model
from app.schemas import CLASS_NAMES, FEATURES, Measurements, Prediction

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "iris_model.pth"
FRONTEND = ROOT / "frontend"


def create_app(model_path: Path = DEFAULT_MODEL_PATH) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        # Her istek için değil, uygulama başlarken yalnızca bir kez yükle.
        application.state.model = load_model(model_path)
        application.state.model_version = hashlib.sha256(model_path.read_bytes()).hexdigest()[:12]
        yield
        application.state.model = None

    application = FastAPI(
        title="Iris Studio API",
        description="Eğitilmiş PyTorch state_dict ile dört cm ölçümünden Iris türü tahmini.",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.mount("/static", StaticFiles(directory=FRONTEND), name="static")

    @application.get("/", include_in_schema=False)
    def index():
        return FileResponse(FRONTEND / "index.html")

    @application.get("/api/health", tags=["Sistem"])
    def health(request: Request):
        ready = getattr(request.app.state, "model", None) is not None
        if not ready:
            raise HTTPException(status_code=503, detail="Model henüz hazır değil.")
        return {"status": "ready", "device": "cpu", "model_version": request.app.state.model_version}

    @application.get("/api/model", tags=["Sistem"])
    def model_info(request: Request):
        model = request.app.state.model
        return {
            "name": "IrisClassifier",
            "architecture": [4, 64, 64, 3],
            "activation": "ReLU",
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "classes": CLASS_NAMES,
            "features": FEATURES,
            "preprocessing": "Ham cm değerleri; ölçeklendirme yok.",
            "probability_note": "Softmax çıktıları kalibre edilmiş doğruluk veya kesinlik ölçümü değildir.",
            "model_version": request.app.state.model_version,
        }

    @application.post("/api/predict", response_model=Prediction, tags=["Tahmin"])
    def predict(measurements: Measurements, request: Request) -> Prediction:
        # Notebook'ta StandardScaler yoktur. Burada da ölçeklendirme uygulama.
        tensor = torch.tensor([measurements.as_vector()], dtype=torch.float32)
        start = perf_counter()
        with torch.inference_mode():
            logits = request.app.state.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
        elapsed = (perf_counter() - start) * 1000
        if not torch.isfinite(probabilities).all():
            raise HTTPException(status_code=500, detail="Model geçerli bir tahmin üretemedi.")
        index = int(probabilities.argmax().item())
        warnings = []
        for feature, value in zip(FEATURES, measurements.as_vector()):
            low, high = feature["observed_range"]
            if not low <= value <= high:
                warnings.append(
                    f'{feature["label"]} ({value:g} cm), notebook\'taki veri aralığının '
                    f"({low:g}–{high:g} cm) dışında. Bu değer için tahmin daha belirsiz olabilir."
                )
        return Prediction(
            request_id=uuid4().hex[:12],
            predicted_at=datetime.now(timezone.utc).isoformat(),
            species=CLASS_NAMES[index],
            class_index=index,
            probability=float(probabilities[index].item()),
            probabilities={name: float(probabilities[i].item()) for i, name in enumerate(CLASS_NAMES)},
            measurements=measurements,
            warnings=warnings,
            inference_ms=round(elapsed, 3),
            model_version=request.app.state.model_version,
        )

    return application


app = create_app()
