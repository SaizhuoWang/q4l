import pickle

import hydra

from q4l.data.loader import Q4LDataLoader
from q4l.utils.log import init_q4l_logging


def test_alpha101():
    init_q4l_logging(is_debug_mode=True)
    with hydra.initialize(config_path="../config", version_base=None):
        cfg = hydra.compose(config_name="test_alpha101")
        loader = Q4LDataLoader(cfg.experiment.data.loader)
        alpha101_df = loader.load(
            instruments="csi1000"
        )  # Load and compute Alpha101 factors for CSI1000
        with open("alpha101.pkl", "wb") as f:
            pickle.dump(alpha101_df, f)


def test_alpha191():
    init_q4l_logging(is_debug_mode=True)
    with hydra.initialize(config_path="../config", version_base=None):
        cfg = hydra.compose(config_name="test_alpha191")
        loader = Q4LDataLoader(cfg.experiment.data.loader)
        alpha101_df = loader.load(
            instruments="csi1000"
        )  # Load and compute Alpha101 factors for CSI1000
        with open("alpha191.pkl", "wb") as f:
            pickle.dump(alpha101_df, f)


import sys

globals()[f"test_alpha{sys.argv[1]}"]()
