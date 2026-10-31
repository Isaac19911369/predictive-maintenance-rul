from .base import RULModel
from .cnn import CNNModel
from .tabular import LightGBMModel, RidgeModel

MODELS: dict[str, type[RULModel]] = {
    "ridge": RidgeModel,
    "lightgbm": LightGBMModel,
    "cnn1d": CNNModel,
}

__all__ = ["MODELS", "CNNModel", "LightGBMModel", "RULModel", "RidgeModel"]
