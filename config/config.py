from typing import List
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
from constants import DatasetType, ModelType, SamplingType
from omegaconf import MISSING, OmegaConf


@dataclass
class Model:
    hyperparameters_fixed: dict = MISSING
    hyperparameters_sweep: dict = MISSING
    type: ModelType = MISSING
    
@dataclass
class MLPLOB(Model):
    hyperparameters_fixed: dict = field(default_factory=lambda: {"num_layers": 3, "hidden_dim": 40, "lr": 0.0003, "seq_size": 384, "all_features": True})
    hyperparameters_sweep: dict = field(default_factory=lambda: {"num_layers": [3, 6], "hidden_dim": [128], "lr": [0.0003], "seq_size": [384]})
    type: ModelType = ModelType.MLPLOB
    
@dataclass
class TLOB(Model):
    hyperparameters_fixed: dict = field(default_factory=lambda: {"num_layers": 4, "hidden_dim": 40, "num_heads": 1, "is_sin_emb": True, "lr": 0.0001, "seq_size": 128, "all_features": True})
    hyperparameters_sweep: dict = field(default_factory=lambda: {"num_layers": [4, 6], "hidden_dim": [128, 256], "num_heads": [1], "is_sin_emb": [True], "lr": [0.0001], "seq_size": [128]})
    type: ModelType = ModelType.TLOB
    
@dataclass
class BiNCTABL(Model):
    hyperparameters_fixed: dict = field(default_factory=lambda: {"lr": 0.001, "seq_size": 10, "all_features": False})
    hyperparameters_sweep: dict = field(default_factory=lambda: {"lr": [0.001], "seq_size": [10]})
    type: ModelType = ModelType.BINCTABL

@dataclass
class DeepLOB(Model):
    hyperparameters_fixed: dict = field(default_factory=lambda: {"lr": 0.01, "seq_size": 100, "all_features": False})
    hyperparameters_sweep: dict = field(default_factory=lambda: {"lr": [0.01], "seq_size": [100]})
    type: ModelType = ModelType.DEEPLOB

@dataclass
class Dataset:  
    type: DatasetType = MISSING
    dates: list = MISSING
    batch_size: int = MISSING

@dataclass
class FI_2010(Dataset):
    type: DatasetType = DatasetType.FI_2010
    dates: list = field(default_factory=lambda: ["2010-01-01", "2010-12-31"])
    batch_size: int = 32

@dataclass
class LOBSTER(Dataset):
    type: DatasetType = DatasetType.LOBSTER
    dates: list = field(default_factory=lambda: ["2015-01-02", "2015-01-30"])
    sampling_type: SamplingType = SamplingType.QUANTITY
    sampling_time: str = "1s"
    sampling_quantity: int = 500
    training_stocks: list = field(default_factory=lambda: ["INTC"])
    testing_stocks: list = field(default_factory=lambda: ["INTC"])
    batch_size: int = 128
    
@dataclass
class BTC(Dataset):
    type: DatasetType = DatasetType.BTC
    dates: list = field(default_factory=lambda: ["2023-01-09", "2023-01-20"])
    sampling_type: SamplingType = SamplingType.NONE
    sampling_time: str = "100ms"
    sampling_quantity: int = 0
    batch_size: int = 128
    training_stocks: list = field(default_factory=lambda: ["BTC"])
    testing_stocks: list = field(default_factory=lambda: ["BTC"])

@dataclass
class Experiment:
    is_data_preprocessed: bool = False
    is_wandb: bool = True
    is_sweep: bool = False
    type: list = field(default_factory=lambda: ["TRAINING"])
    is_debug: bool = False
    checkpoint_reference: str = ""
    seed: int = 1
    horizon: int = 10
    max_epochs: int = 10
    dir_ckpt: str = "model.ckpt"
    optimizer: str = "Adam"
    
defaults = [Model, Experiment, Dataset]

@dataclass
class Config:
    model: Model
    dataset: Dataset
    experiment: Experiment = field(default_factory=Experiment)
    defaults: List = field(default_factory=lambda: [
        {"hydra/job_logging": "disabled"},
        {"hydra/hydra_logging": "disabled"},
        "_self_"
    ])
    
cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="model", name="mlplob", node=MLPLOB)
cs.store(group="model", name="tlob", node=TLOB)
cs.store(group="model", name="binctabl", node=BiNCTABL)
cs.store(group="model", name="deeplob", node=DeepLOB)
cs.store(group="dataset", name="lobster", node=LOBSTER)
cs.store(group="dataset", name="fi_2010", node=FI_2010)
cs.store(group="dataset", name="btc", node=BTC)