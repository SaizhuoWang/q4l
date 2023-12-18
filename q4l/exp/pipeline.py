import os
import typing
from typing import Dict

import yaml
from lightning.pytorch import Trainer
from omegaconf import OmegaConf

from ..config import ExperimentConfig, JobConfig
from ..data.dataset import DataModuleWrapper, Q4LDataset
from ..qlib import init as qlib_init
from ..qlib.model.base import Model
from ..qlib.utils import fill_placeholder
from ..qlib.workflow import R
from ..qlib.workflow.recorder import MLflowRecorder, Recorder
from ..utils.log import LoggingContext, get_logger
from ..utils.misc import create_instance, make_qlib_init_config


def has_ckpt(run_dir: str):
    return os.path.exists(os.path.join(run_dir, "model_ckpt.pkl"))


def train(cfg: ExperimentConfig):
    rec = R.get_recorder()
    logger = get_logger("my_train")

    logger.info(f"Loading dataset")
    dataset = Q4LDataset(cfg)
    data_module = DataModuleWrapper(dataset)
    logger.info(f"Loaded dataset. Dataset information:\n{dataset.digest()}")

    # this dataset is saved for online inference (as you may see below).
    # So the concrete data should not be dumped
    logger.info("Saving dataset information")
    dataset.config(dump_all=False, recursive=True)
    rec.save_objects(**{"dataset": dataset}, artifact_path=R.suffix)

    logger.info(f"Rewriting model config according to dataset")
    new_cfg = dataset.rewrite_model_config(cfg)

    logger.info("Creating model")
    model: Model = quant_model_factory(new_cfg)

    logger.info("Model created. Creating trainer")
    trainer = Trainer()

    # logger.info("Creating data loader for train, valid and test")
    # train_loader, valid_loader, test_loader = dataset.make_dataloader()

    # NOTE: reweighter is not supported in this version
    # reweighter: Reweighter = cfg.get("reweighter", None)

    checkpoint_dir = os.path.join(rec.get_artifact_uri(), R.suffix)
    if has_ckpt(checkpoint_dir):
        logger.info("resume training from checkpoint in ")
        model.load_checkpoint(checkpoint_dir)

    # model training
    logger.info("Start training")
    trainer.fit(model, data_module)
    rec.save_objects(**{"model.pkl": model}, artifact_path=R.suffix)

    # Test model performance
    logger.info("Start testing")
    trainer.test(model, data_module)

    # Make inferences
    logger.info("Start making inferences")
    pred = trainer.predict(model, data_module)

    # fill placehorder for later inference
    logger.info("Filling placeholder in recorder configs for later inference")
    placehorder_value = {
        "<MODEL>": model,
        "<DATASET>": dataset,
        "<TRAINER>": trainer,
        "<PRED>": pred,
    }
    recorder_cfg: Dict = fill_placeholder(
        OmegaConf.to_container(cfg.collector), placehorder_value
    )

    # Record the performance
    logger.info("Start recording performance")
    for record in recorder_cfg["single"]:
        # Some recorder require the parameter `model` and `dataset`.
        # try to automatically pass in them to the initialization function
        # to make defining the tasking easier
        logger.info(f"Running recorder {record}")
        r = create_instance(
            record,
            default_module="qlib.workflow.record_temp",
            try_kwargs={"model": model, "dataset": dataset},
            recorder=rec,
        )
        r.generate()


# @contextlib.contextmanager
def q4l_task_wrapper_fn(
    exp_cfg: ExperimentConfig,
    job_cfg: JobConfig,
    repeat_index: int = 0,
    is_subprocess: bool = False,
    actual_func: typing.Callable = train,
    **kwargs,  # Other parameters that will be passed to the actual function
) -> Recorder:
    """
    A wrapper function for setting up mlflow tracking and saving current config:
        1. Init qlib again if necessary (in subprocess)
        2. Setup mlflow logging
        3. Redirect logging to mlflow, recorder suffix is set to `repeat_index`/`test_interval`
        4. Set recorder prefix
    Upon exiting, the logging will be redirected back to the original logger

    Returns
    ----------
    Recorder: The instance of the recorder
    """
    # If I am run by a subprocess, I should init qlib again
    if is_subprocess:
        qlib_init(
            **make_qlib_init_config(exp_config=exp_cfg, job_config=job_cfg)
        )
    # Attach suffix to the Singleton instance `R`, mlflow setup
    test_interval = exp_cfg.time.segments.test[0]
    logdir_suffix = f"{repeat_index}/{test_interval.start}~{test_interval.end}"
    R.set_suffix(logdir_suffix)
    R.set(
        experiment_name=job_cfg.name.exp_name,
        recorder_name=job_cfg.name.run_name,
    )
    recorder: MLflowRecorder = R.get_recorder(
        experiment_name=job_cfg.name.exp_name,
        recorder_name=job_cfg.name.run_name,
    )
    logdir = os.path.join(recorder.get_artifact_uri(), logdir_suffix)
    os.makedirs(logdir, exist_ok=True)
    # Dump current rolling run's config
    with open(os.path.join(logdir, "exp_config.yaml"), "w") as f:
        yaml.safe_dump(
            OmegaConf.to_container(exp_cfg, resolve=True), f, indent=2
        )
    # Logging redirection
    with LoggingContext(is_debug=job_cfg.misc.debug, recorder_wrapper=R):
        logger = get_logger("pipeline")
        logger.info("Finished setup mlflow logging. Start running...")
        # return None
        # Do the actual training
        func_kwargs = {
            "exp_config": exp_cfg,
            "job_config": job_cfg,
            "repeat_index": repeat_index,
            "is_subprocess": is_subprocess,
            **kwargs,
        }
        actual_func(**func_kwargs)
    # Post-task recovery
    R.set_suffix(str(repeat_index))
    return recorder
