# Config schema for experiments and SLURM jobs
import typing as tp
from ast import Dict
from dataclasses import dataclass
from datetime import datetime

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf


@dataclass
class NameConfig:
    exp_name: str = "q4l_default_exp"
    run_name: str = "q4l_default_run"
    slurm_jobname: str = "qrs"


@dataclass
class ResourceConfig:
    cpu_per_task: int = 4
    total_gpu: int = 1


@dataclass
class RepeatConfig:
    total_repeat: int = 1
    split_repeat: bool = False
    repeat_per_job: int = 1


@dataclass
class ParallelConfig:
    repeat: int = 1
    rolling: int = 1


@dataclass
class MachineConfig:
    data_root: str
    log_root: str
    factor_cache_dir: str
    taskdb_uri: str
    taskdb_name: str
    mlflow_tracking_uri: str
    tensorboard_dir: str


@dataclass
class MiscConfig:
    device: str
    timestamp: str
    prepare_shm: bool = False
    debug: bool = False  # Debug mode flag
    seed: int = 0  # Random seed for reproducibility
    model_name: str = "default"
    num_nodes: int = 1
    mongo_job_id: str = "N/A"
    use_disk_cache: bool = True
    refresh_disk_cache: bool = False


@dataclass
class JobConfig:
    name: NameConfig
    resource: ResourceConfig
    repeat: RepeatConfig
    parallel: ParallelConfig
    machine: MachineConfig
    misc: MiscConfig


@dataclass
class ModuleConfig:
    name: str
    module_path: str
    kwargs: tp.Dict


@dataclass
class BackendConfig:
    storage: tp.Dict[
        str, ModuleConfig
    ]  # Storage backend (e.g. {"disk": DiskStorage, "mongodb": MongoDBStorage})
    compute: tp.Dict[
        str, ModuleConfig
    ]  # Compute backend (e.g. {"numpy": NumpyCompute, "hxdf": HXDFCompute})


@dataclass
class AlphaGroupConfig:
    name: str  # Alpha group name (e.g. Alpha101, Volume-price_1, etc.)
    compute_backend: str  # Computing backend for this alpha group (e.g. "hxdf", "numpy", etc.)
    expressions: tp.Dict[
        str, str
    ]  # Alpha expressions (e.g. {"alpha1": "Rank(Correlation(vwap, volume, 5), 10)"})
    # For alpha expression syntax, please refer to q4l doc.
    # Alpha expressions can be written in the YAML config, or loaded from a YAML file.


@dataclass
class AlphaConfig:
    x: tp.Dict[str, AlphaGroupConfig]
    y: tp.Dict[str, AlphaGroupConfig]


@dataclass
class LoaderConfig:
    backend: BackendConfig  # Specification of storage backend and computing backend
    alpha: tp.Dict[str, AlphaGroupConfig]  # A list of alpha groups
    default_compute_backend: str = (
        "numpy"  # Default compute backend for all alpha groups
    )
    verbose: bool = False  # Whether to print verbose info during loading


@dataclass
class PreprocessorConfig:  # Data pre-processing after loading from disk
    learn: tp.List[
        ModuleConfig
    ]  # Pre-processing modules for training data  (may contain future-info)
    infer: tp.List[
        ModuleConfig
    ]  # Pre-processing modules for inference data (no future-info leakage is allowed)
    shared: tp.List[ModuleConfig]  # Shared pre-processing modules


@dataclass
class SamplerConfig:
    batch_size: int  # Batch size for batch sampling
    x_window: int  # window size for x
    y_window: int  # window size for y
    sample_mode: str  # slicing mode (e.g. 'C' for cross-sectional or 'T' for time-series)
    filters: tp.Dict[
        str, ModuleConfig
    ]  # Sample filter (e.g. filter out samples with too many NaNs)
    # Filters are applied to the dataset sequentially in the order of the list.
    x_group: tp.List[str]
    y_group: tp.List[str]
    keep_group: tp.List[str]
    sample_y_as_x: bool  # Whether to sample y as x in the same way


@dataclass
class DataConfig:  # Data loading is essentially an ETL process
    loader: LoaderConfig  # Specifies "external_storage" => "in-memory raw data"  (The Extract part of ETL)
    preprocessor: PreprocessorConfig  # Specifies "in-memory raw data" => "in-memory pre-processed data"  (The Transform part of ETL)
    sampler: SamplerConfig  # Specifies "in-memory pre-processed data" => "in-memory sampled data"  (The Load part of ETL)

    # Some misc configs
    pool: str  # Stock/Instrument universe (e.g. "csi500", "hsi", "sp500", etc.)
    region: str  # Region (e.g. "cn", "hk", "us", etc.)
    # Shared memory configs
    use_shm: bool  # Whether to use shared memory (for memory saving in parallel computing)
    shm_name: str  # Name of the shared memory file in /dev/shm
    # Disk cache config
    fetch_orig: bool  # Whether to fetch original data instead of disk cache (e.g. for debugging)
    benchmark: str  # Backtest benchmark ticker code (e.g. "000300.SH", "^GSPC", etc.)
    graph: tp.Dict  # Graph data config


@dataclass
class TimeInterval:
    start: tp.Any
    end: tp.Any


@dataclass
class SegmentConfig:
    train: tp.List[TimeInterval]
    valid: tp.List[TimeInterval]
    test: tp.List[TimeInterval]


@dataclass
class TimeConfig:  # All time-related configs
    # Related to the Extract part of ETL
    start_time: str  # Start time of the whole experiment
    end_time: str  # End time of the whole experiment

    # Related to the Transform part of ETL
    fit_start_time: str  # Start time of the training period
    fit_end_time: str  # End time of the training period (Start of the validation period)

    # Related to the Load part of ETL
    segments: SegmentConfig  # Train-valid-test split of the outer loop
    rolling_step: int  # Rolling step (e.g. 1, 5, 10, etc.) for splitting the whole testing period into multiple rolling tasks
    rolling_type: str  # Mode of rolling, "sliding"/"expanding"/"only_new"/"throw_away"


@dataclass
class MetricConfig:
    validation: ModuleConfig  # Validation metrics
    testing: ModuleConfig  # Test metrics


@dataclass
class ModelConfig:
    input_size: int
    basic_info: tp.Dict  # Basic info of the model (e.g. input_dim, output_dim, gpu_index, etc.)
    model_type: str  # Model type (e.g. "temporal", "spatiotemporal", etc.)
    output_type: str  # Output type ("signal" or "position")
    name: str  # Model name (e.g. "lstm", "transformer", etc.)
    components: tp.Dict[
        str, ModuleConfig
    ]  # Model components (e.g. "encoder", "decoder", etc.)
    optimizer: tp.Dict  # TODO: Optimizer too complex, use a versatile dict for now
    trainer: tp.Dict  # Training configs (e.g. "epochs", "batch_size", "early_stopping", etc.). These configs are passed to the initialization of the trainer
    loss: ModuleConfig  # Loss function, subclass of nn.Module
    metric: MetricConfig  # Evaluation metrics, refer to torchmetrics


@dataclass
class CollectorConfig:  # mlflow-based tracking and evaluation.
    single: tp.Dict[
        str, tp.Any
    ]  # Evaluation for each single atomic task (e.g. a rolling)
    rolling: tp.Dict[str, tp.Any]  # Evaluation for a full rolling task
    repeat: tp.Dict[str, tp.Any]  # Evaluation for a full repeated task
    zoo: tp.Dict[str, tp.Any]  # A zoo for repeatedly used configs


@dataclass
class TradingConfig:  # Order execution config
    pass


@dataclass
class ExperimentConfig:
    data: DataConfig  # Everything related to data
    time: TimeConfig  # Dataset splitting, rolling, and other time-related stuff
    model: ModelConfig  # Model architecture, optimizer, loss, and other model-related stuff
    strategy: ModuleConfig  # Portfolio strategy, order execution, and other trading-related stuff
    # trading: TradingConfig  # Order execution config
    collector: CollectorConfig  # Evaluation metrics, recorders, and other evaluation-related stuff


@dataclass
class GlobalConfig:
    experiment: ExperimentConfig
    job: JobConfig


# --------- Register config schemas upon import --------- #
cs = ConfigStore.instance()

# Register structural config schemas
cs.store(name="base_job", node=JobConfig, group="job")
cs.store(name="base_name", node=NameConfig, group="job/name")
cs.store(name="base_resource", node=ResourceConfig, group="job/resource")
cs.store(name="base_repeat", node=RepeatConfig, group="job/repeat")
cs.store(name="base_parallel", node=ParallelConfig, group="job/parallel")
cs.store(name="base_machine", node=MachineConfig, group="job/machine")
cs.store(name="base_misc", node=MiscConfig, group="job/misc")

# Register experiment config schemas
cs.store(name="base_experiment", node=ExperimentConfig, group="experiment")
cs.store(name="base_data", node=DataConfig, group="experiment/data")
cs.store(
    name="base_data_loader", node=LoaderConfig, group="experiment/data/loader"
)
cs.store(
    name="base_data_loader_backend",
    node=BackendConfig,
    group="experiment/data/loader/backend",
)
cs.store(
    name="base_data_loader_alpha",
    node=AlphaGroupConfig,
    group="experiment/data/loader/alpha",
)
cs.store(
    name="base_data_preprocessor",
    node=PreprocessorConfig,
    group="experiment/data/preprocessor",
)
cs.store(
    name="base_data_sampler",
    node=SamplerConfig,
    group="experiment/data/sampler",
)
cs.store(name="base_time", node=TimeConfig, group="experiment/time")
cs.store(name="base_sample", node=SamplerConfig, group="experiment/sample")
cs.store(name="base_model", node=ModelConfig, group="experiment/model")
cs.store(
    name="base_data_preprocessor",
    node=PreprocessorConfig,
    group="experiment/data/preprocessor",
)
cs.store(name="base_metric", node=MetricConfig, group="experiment/model/metric")
# cs.store(name="base_strategy", node=ModuleConfig, group="experiment/strategy")
cs.store(
    name="base_collector", node=CollectorConfig, group="experiment/collector"
)

# Register global config
cs.store(name="base_config", node=GlobalConfig)


# ------------------------------------------------------- #

# Some utility functions
def hydra_datetime_formatter(dt: datetime) -> str:
    return f"${{timestamp:{dt.isoformat()}}}"


def create_timeinterval_from_datetime(
    start: datetime, end: datetime
) -> TimeInterval:
    return TimeInterval(
        start=hydra_datetime_formatter(start), end=hydra_datetime_formatter(end)
    )


def visualize():
    with hydra.initialize(config_path="q4l_builtin"):
        cfg = hydra.compose(config_name="q4l_global_default")
        OmegaConf.save(cfg, f="q4l_global_default.yaml")


GLOBAL_CONFIG: Dict = {}


def register_global_config(config: GlobalConfig):
    GLOBAL_CONFIG["cfg"] = config
