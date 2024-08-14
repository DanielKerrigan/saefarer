"""Code for training a sparse autoencoder."""

import json
import time
from dataclasses import asdict
from os import PathLike
from pathlib import Path
from typing import TypedDict, Union

import torch
import tqdm
import wandb
from datasets import (
    Dataset,
    IterableDataset,
)
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedModel,
)

from saefarer.activations_store import ActivationsStore
from saefarer.config import TrainingConfig
from saefarer.model import SAE, ForwardOutput


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


def train(
    cfg: TrainingConfig,
    model: PreTrainedModel,
    dataset: Union[Dataset, IterableDataset, DataLoader],
    save_path: Union[str, PathLike],
    log_path: Union[str, PathLike],
) -> SAE:
    """Train the SAE"""

    log_path = Path(log_path)

    if cfg.logger == "wandb":
        wandb.init(
            config=asdict(cfg),
            project=cfg.wandb_project,
            group=cfg.wandb_group,
            name=cfg.wandb_name,
            notes=cfg.wandb_notes,
            dir=log_path,
        )
    else:
        log_file = log_path.open("a")

    sae = SAE(cfg)

    store = ActivationsStore(model, dataset, cfg)

    optimizer = torch.optim.Adam(
        sae.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps
    )

    print("Beginning training")

    start_time = time.time()

    for i in tqdm.trange(1, cfg.total_training_batches + 1):
        # get next batch of model activations
        x = store.next_batch()

        # forward pass through SAE
        output: ForwardOutput = sae(x)

        # backward pass

        output.loss.backward()

        sae.set_decoder_norm_to_unit_norm()
        sae.remove_gradient_parallel_to_decoder_directions()

        optimizer.step()
        optimizer.zero_grad()

        # logging

        if i % cfg.log_batch_freq == 0:
            info = LogData(
                elapsed_seconds=time.time() - start_time,
                n_training_batches=i,
                n_training_tokens=i * sae.cfg.sae_batch_size_tokens,
                loss=output.loss.item(),
                mse_loss=output.mse_loss.item(),
                aux_loss=output.aux_loss.item(),
                n_dead_features=output.num_dead,
                mean_n_batches_since_fired=sae.stats_last_nonzero.mean(
                    dtype=torch.float32
                ).item(),
                max_n_batches_since_fired=int(sae.stats_last_nonzero.max().item()),
            )

            json_line = json.dumps(info)

            print(json_line)

            if cfg.logger == "wandb":
                wandb.log(data=info, step=i)
            else:
                log_file.write(json_line + "\n")

    print("Saving final model")

    sae.save(save_path)

    if cfg.logger == "wandb":
        wandb.finish()
    else:
        log_file.close()

    return sae
