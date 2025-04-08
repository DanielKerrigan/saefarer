"""Code for training a sparse autoencoder."""

import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import tqdm
from torch.utils.data import DataLoader

from saefarer.sae import SAE
from saefarer.training import logger
from saefarer.training.activations_store import ActivationsStore

if TYPE_CHECKING:
    from os import PathLike

    from datasets import (
        Dataset,
        IterableDataset,
    )
    from transformers import PreTrainedModel

    from saefarer.sae import ForwardOutput
    from saefarer.training.config import TrainingConfig


def train(
    cfg: "TrainingConfig",
    model: "PreTrainedModel",
    dataset: "Dataset | IterableDataset | DataLoader",
    save_path: "str | PathLike",
    log_dir: "str | PathLike",
    checkpoint_dir: "str | PathLike | None" = None,
) -> SAE:
    """Train the SAE"""

    log = logger.from_cfg(cfg, log_dir)

    sae = SAE(cfg)

    store = ActivationsStore(model, dataset, cfg)

    optimizer = torch.optim.Adam(
        sae.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
    )

    print("Beginning training")

    start_time = time.time()

    for i in tqdm.trange(
        1, cfg.total_training_batches + 1, disable=not cfg.show_progress
    ):
        # get next batch of model activations
        x = store.next_batch()

        # forward pass through SAE
        output: "ForwardOutput" = sae(x)

        # backward pass

        output.loss.backward()

        sae.set_decoder_norm_to_unit_norm()
        sae.remove_gradient_parallel_to_decoder_directions()

        optimizer.step()
        optimizer.zero_grad()

        # logging

        if cfg.log_batch_freq and i % cfg.log_batch_freq == 0:
            log_data = logger.LogData(
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

            log.write(log_data)

        if (
            cfg.checkpoint_batch_freq
            and checkpoint_dir
            and i % cfg.checkpoint_batch_freq == 0
        ):
            print(f"Saving checkpoint after batch {i}")
            sae.save(Path(checkpoint_dir) / f"batch_{i}.pt")

    print("Saving final model")

    sae.save(save_path)

    log.close()

    return sae
