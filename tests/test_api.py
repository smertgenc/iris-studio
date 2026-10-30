"""Gerçek yüklenmiş ağırlıklarla API, sınıf sırası ve giriş doğrulama testleri."""

from fastapi.testclient import TestClient
import pytest
import torch

from app.main import DEFAULT_MODEL_PATH, app, create_app
from app.model import IrisClassifier
from app.schemas import CLASS_NAMES, Measurements


SAMPLES = [
    ([5.1, 3.5, 1.4, 0.2], "Iris-setosa"),
    ([6.0, 2.9, 4.5, 1.5], "Iris-versicolor"),
    ([6.5, 3.0, 5.8, 2.2], "Iris-virginica"),
]
KEYS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize("values,expected", SAMPLES)
def test_actual_model_predictions(client, values, expected):
    response = client.post("/api/predict", json=dict(zip(KEYS, values)))
    assert response.status_code == 200
    data = response.json()
    assert data["species"] == expected
    assert data["class_index"] == CLASS_NAMES.index(expected)
    assert sum(data["probabilities"].values()) == pytest.approx(1, abs=1e-6)
    assert data["probability"] == max(data["probabilities"].values())
    assert data["warnings"] == []


def test_notebook_equivalence_and_feature_order(client):
    # JSON anahtarları ters sırada gelse de notebook ile aynı sonucu üretmeli.
    values = [6.2, 2.8, 4.8, 1.8]
    payload = dict(reversed(list(zip(KEYS, values))))
    result = client.post("/api/predict", json=payload).json()
    notebook_model = IrisClassifier()
    notebook_model.load_state_dict(torch.load(DEFAULT_MODEL_PATH, map_location="cpu", weights_only=True))
    notebook_model.eval()
    with torch.inference_mode():
        expected = torch.softmax(notebook_model(torch.tensor([values], dtype=torch.float32)), dim=1)[0]
    assert [result["probabilities"][s] for s in CLASS_NAMES] == pytest.approx(expected.tolist(), abs=1e-7)


@pytest.mark.parametrize("change", [
    {"sepal_length": -1}, {"sepal_width": 0}, {"petal_width": 5},
    {"petal_length": "4.5"}, {"sepal_length": True}, {"Id": 1},
    {"petal_length": None}, {"sepal_width": 100000},
])
def test_rejects_bad_measurements(client, change):
    payload = dict(zip(KEYS, SAMPLES[0][0])) | change
    assert client.post("/api/predict", json=payload).status_code == 422


def test_rejects_missing_fields(client):
    assert client.post("/api/predict", json={"sepal_length": 5.1}).status_code == 422


def test_rejects_non_finite_numbers():
    for value in [float("nan"), float("inf"), float("-inf")]:
        with pytest.raises(ValueError):
            Measurements(sepal_length=value, sepal_width=3.5, petal_length=1.4, petal_width=0.2)


def test_outside_observed_range_is_reported_without_clipping(client):
    payload = dict(zip(KEYS, [9.0, 3.5, 1.4, 0.2]))
    result = client.post("/api/predict", json=payload).json()
    assert result["measurements"]["sepal_length"] == 9.0
    assert len(result["warnings"]) == 1
    assert "aralığının" in result["warnings"][0]


def test_site_and_model_metadata(client):
    assert client.get("/").status_code == 200
    assert "Iris Studio" in client.get("/").text
    for asset in ["styles.css", "app.js", "assets/favicon.svg"]:
        assert client.get(f"/static/{asset}").status_code == 200
    assert client.get("/api/health").json()["status"] == "ready"
    info = client.get("/api/model").json()
    assert info["architecture"] == [4, 64, 64, 3]
    assert info["parameter_count"] == 4675
    assert info["classes"] == list(CLASS_NAMES)
    assert client.get("/openapi.json").status_code == 200
    # Checkpoint statik dosya olarak dışarı sunulmuyor.
    assert client.get("/static/../models/iris_model.pth").status_code == 404


def test_missing_checkpoint_fails_at_startup(tmp_path):
    with pytest.raises(RuntimeError, match="Model dosyası bulunamadı"):
        with TestClient(create_app(tmp_path / "missing.pth")):
            pass


def test_incompatible_checkpoint_fails_at_startup(tmp_path):
    path = tmp_path / "wrong.pth"
    torch.save({"wrong_layer.weight": torch.zeros(1)}, path)
    with pytest.raises(RuntimeError, match="Model yüklenemedi"):
        with TestClient(create_app(path)):
            pass
