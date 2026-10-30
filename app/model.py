"""Notebook'un 21. hücresindeki mimari; state_dict anahtarları aynıdır."""

from pathlib import Path

import torch
from torch import nn


class IrisClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=4, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer_stack(x)


def load_model(path: Path) -> IrisClassifier:
    if not path.is_file():
        raise RuntimeError(
            f"Model dosyası bulunamadı: {path}. "
            "Eğittiğiniz state_dict dosyasını models/iris_model.pth konumuna koyun."
        )
    model = IrisClassifier()
    try:
        # weights_only=True: yalnızca ağırlıkları oku; GPU gerektirme.
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise RuntimeError(
            "Model yüklenemedi. Dosya, notebook'taki linear_layer_stack "
            "adlı 4→64→64→3 ağının state_dict çıktısı olmalıdır."
        ) from exc
    model.eval()
    model.requires_grad_(False)
    return model
