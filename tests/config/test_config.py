from hydra import compose, initialize
from omegaconf import OmegaConf


def test_config():
    with initialize(config_path=".", version_base=None):
        cfg = compose("base")
    with open("base_config_demo.yaml", "w") as f:
        OmegaConf.save(cfg, f)
