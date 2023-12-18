import pandas as pd
import qlib
from hydra import compose, initialize
from omegaconf import OmegaConf
from qlib.workflow import R
from qlib.workflow.record_temp import PortAnaRecord

from q4l.config import GlobalConfig, ModuleConfig, register_global_config
from q4l.utils.misc import fill_backtest_config, make_qlib_init_config


def test_q4l_xchg():
    with initialize(config_path="../config", version_base=None):
        cfg: GlobalConfig = compose(config_name="base")
        register_global_config(cfg)
        qlib.init(
            **make_qlib_init_config(
                exp_config=cfg.experiment, job_config=cfg.job
            )
        )
        with R.start(experiment_name="test_backtest", recorder_name="test1"):
            # Compose a test config
            recorder = R.get_recorder()
            xchg_cfg: ModuleConfig = cfg.experiment.collector.rolling
            transformed_xchg_cfg = fill_backtest_config(
                OmegaConf.to_container(xchg_cfg, resolve=True),
                cfg.experiment,
                cfg.job,
            )

            # Create a fake prediction
            fake_pred = pd.read_csv(
                "/wsz/Data/my_data_dir/main/jp/features/day/close.csv",
                index_col=0,
            )
            fake_pred.index = pd.to_datetime(fake_pred.index)
            fake_pred = fake_pred.stack(dropna=False)
            fake_pred.index.set_names(["datetime", "instrument"], inplace=True)
            fake_pred.sort_index(inplace=True)

            PortAnaRecord(
                config=transformed_xchg_cfg["portfolio_analysis"]["kwargs"][
                    "config"
                ],
                recorder=recorder,
            )._generate(pred=fake_pred)


test_q4l_xchg()
