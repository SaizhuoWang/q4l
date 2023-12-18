from typing import Any

from lightning.pytorch.utilities.types import STEP_OUTPUT

from q4l.config import ExperimentConfig

from ....base import RLPortfolioModel


class DeepTrader(RLPortfolioModel):
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)

    def build_model_arch(self):
        return super().build_model_arch()

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return super().training_step(*args, **kwargs)
