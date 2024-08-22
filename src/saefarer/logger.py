import json
from dataclasses import asdict
from importlib import import_module
from os import PathLike
from pathlib import Path
from typing import Union

from torch.utils.tensorboard.writer import SummaryWriter

from saefarer.config import TrainingConfig
from saefarer.types import LogData


class Logger:
    def __init__(self, cfg: TrainingConfig, log_path: Union[str, PathLike]):
        self.cfg = cfg
        self.log_path = Path(log_path)

    def write(self, data: LogData):
        pass

    def close(self):
        pass


class WAndBLogger(Logger):
    def __init__(self, cfg: TrainingConfig, log_path: Union[str, PathLike]):
        super().__init__(cfg, log_path)

        self.wandb = import_module("wandb")

        self.wandb.init(
            config=asdict(cfg),
            project=cfg.wandb_project,
            group=cfg.wandb_group,
            name=cfg.wandb_name,
            notes=cfg.wandb_notes,
            dir=self.log_path,
        )

    def write(self, data: LogData):
        self.wandb.log(data=data, step=data["n_training_batches"])

    def close(self):
        self.wandb.finish()


class TensorboardLogger(Logger):
    def __init__(self, cfg: TrainingConfig, log_path: Union[str, PathLike]):
        super().__init__(cfg, log_path)
        self.writer = SummaryWriter(self.log_path)

    def write(self, data: LogData):
        for [key, value] in data.items():
            self.writer.add_scalar(key, value, global_step=data["n_training_batches"])
        self.writer.flush()

    def close(self):
        self.writer.close()


class JSONLLogger(Logger):
    def __init__(self, cfg: TrainingConfig, log_path: Union[str, PathLike]):
        super().__init__(cfg, log_path)
        self.log_file = self.log_path.open("a")

    def write(self, data: LogData):
        json_line = json.dumps(data)
        self.log_file.write(json_line + "\n")

    def close(self):
        self.log_file.close()


def from_cfg(cfg: TrainingConfig, log_path: Union[str, PathLike]) -> Logger:
    if cfg.logger == "jsonl":
        return JSONLLogger(cfg, log_path)
    elif cfg.logger == "tensorboard":
        return TensorboardLogger(cfg, log_path)
    else:
        return WAndBLogger(cfg, log_path)
