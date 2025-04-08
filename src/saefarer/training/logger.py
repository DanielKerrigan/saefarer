import json
from dataclasses import asdict
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from os import PathLike

    from saefarer.training.config import TrainingConfig


class LogData(TypedDict):
    elapsed_seconds: float
    n_training_batches: int
    n_training_tokens: int
    loss: float
    mse_loss: float
    aux_loss: float
    n_dead_features: int
    mean_n_batches_since_fired: float
    max_n_batches_since_fired: int


class Logger:
    def __init__(self, cfg: "TrainingConfig", log_dir: "str | PathLike"):
        self.cfg = cfg
        self.log_dir = Path(log_dir)

    def write(self, data: LogData):
        pass

    def close(self):
        pass


class WAndBLogger(Logger):
    def __init__(self, cfg: "TrainingConfig", log_dir: "str | PathLike"):
        super().__init__(cfg, log_dir)

        self.wandb = import_module("wandb")

        self.wandb.init(
            config=asdict(cfg),
            project=cfg.wandb_project,
            group=cfg.wandb_group,
            name=cfg.wandb_name,
            notes=cfg.wandb_notes,
            dir=self.log_dir,
        )

    def write(self, data: LogData):
        self.wandb.log(data=data, step=data["n_training_batches"])

    def close(self):
        self.wandb.finish()


class TensorboardLogger(Logger):
    def __init__(self, cfg: "TrainingConfig", log_dir: "str | PathLike"):
        super().__init__(cfg, log_dir)

        from torch.utils.tensorboard.writer import SummaryWriter

        self.writer = SummaryWriter(self.log_dir)

    def write(self, data: LogData):
        for [key, value] in data.items():
            self.writer.add_scalar(key, value, global_step=data["n_training_batches"])
        self.writer.flush()

    def close(self):
        self.writer.close()


class JSONLLogger(Logger):
    def __init__(self, cfg: "TrainingConfig", log_dir: "str | PathLike"):
        super().__init__(cfg, log_dir)
        self.log_file = (self.log_dir / "logs.jsonl").open("a")

    def write(self, data: LogData):
        json_line = json.dumps(data)
        self.log_file.write(json_line + "\n")

    def close(self):
        self.log_file.close()


def from_cfg(cfg: "TrainingConfig", log_dir: "str | PathLike") -> Logger:
    if cfg.logger == "jsonl":
        return JSONLLogger(cfg, log_dir)
    elif cfg.logger == "tensorboard":
        return TensorboardLogger(cfg, log_dir)
    else:
        return WAndBLogger(cfg, log_dir)
