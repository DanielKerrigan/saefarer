"""Code for training a sparse autoencoder."""

import json
from os import PathLike
from pathlib import Path
from typing import List, Tuple, TypedDict, Union

import torch
import tqdm
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
)
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from saefarer.activations_store import ActivationsStore
from saefarer.config import Config
from saefarer.model import SAE, ForwardOutput


class LogData(TypedDict):
    batch: int
    loss: float
    mse_loss: float
    aux_loss: float
    num_dead: int


def train(
    cfg: Config,
    model: PreTrainedModel,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    dataset: Union[Dataset, DatasetDict, IterableDatasetDict, IterableDataset],
    log_batch_freq: int,
    save_path: Union[str, PathLike],
    log_path: Union[str, PathLike],
) -> Tuple[SAE, List[LogData]]:
    """Train the SAE"""

    sae = SAE(cfg)

    if isinstance(dataset, DatasetDict) or isinstance(dataset, IterableDatasetDict):
        dataset = dataset["train"]

    print("Initializing ActivationsStore")

    store = ActivationsStore(model, tokenizer, dataset, cfg)

    optimizer = torch.optim.Adam(
        sae.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps
    )

    log_data: List[LogData] = []

    print("Beginning training")

    for i in tqdm.trange(cfg.total_training_batches):
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

        if i % log_batch_freq == 0:
            info = LogData(
                batch=i,
                loss=output.loss.item(),
                mse_loss=output.mse_loss.item(),
                aux_loss=output.aux_loss.item(),
                num_dead=output.num_dead,
            )
            log_data.append(info)
            print(info)

    print("Saving final model")

    sae.save(save_path)

    Path(log_path).write_text(json.dumps(log_data), encoding="utf-8")

    return sae, log_data
