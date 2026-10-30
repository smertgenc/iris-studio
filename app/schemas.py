"""API sözleşmesi ve notebook'tan alınan özellik/sınıf sırası."""

from pydantic import BaseModel, ConfigDict, Field


# LabelEncoder.classes_ alfabetik sırası; model çıktısının sırası değişmemeli.
CLASS_NAMES = ("Iris-setosa", "Iris-versicolor", "Iris-virginica")

# observed_range: notebook df.describe() çıktısı. input_range: uygulama sınırı.
FEATURES = (
    {"key": "sepal_length", "label": "Çanak yaprak uzunluğu", "column": "SepalLengthCm", "observed_range": [4.3, 7.9], "input_range": [0.1, 10.0]},
    {"key": "sepal_width", "label": "Çanak yaprak genişliği", "column": "SepalWidthCm", "observed_range": [2.0, 4.4], "input_range": [0.1, 6.0]},
    {"key": "petal_length", "label": "Taç yaprak uzunluğu", "column": "PetalLengthCm", "observed_range": [1.0, 6.9], "input_range": [0.1, 10.0]},
    {"key": "petal_width", "label": "Taç yaprak genişliği", "column": "PetalWidthCm", "observed_range": [0.1, 2.5], "input_range": [0.1, 4.0]},
)


class Measurements(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    sepal_length: float = Field(ge=0.1, le=10, description="Çanak yaprak uzunluğu (cm)")
    sepal_width: float = Field(ge=0.1, le=6, description="Çanak yaprak genişliği (cm)")
    petal_length: float = Field(ge=0.1, le=10, description="Taç yaprak uzunluğu (cm)")
    petal_width: float = Field(ge=0.1, le=4, description="Taç yaprak genişliği (cm)")

    def as_vector(self) -> list[float]:
        # JSON anahtar sırasına güvenme; eğitimdeki sırayı açıkça koru.
        return [self.sepal_length, self.sepal_width, self.petal_length, self.petal_width]


class Prediction(BaseModel):
    request_id: str
    predicted_at: str
    species: str
    class_index: int
    probability: float
    probabilities: dict[str, float]
    measurements: Measurements
    warnings: list[str]
    inference_ms: float
    model_version: str
