"""Code for training a sparse autoencoder."""

from os import PathLike
from typing import Union

import torch
import torch.nn.functional as F
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

from sparse_autoencoder.activations_store import ActivationsStore
from sparse_autoencoder.config import Config
from sparse_autoencoder.model import SAE


def mse(output, target):
    """MSE loss"""
    return F.mse_loss(output, target, reduction="mean")


def normalized_mse(output, target):
    """Normalized MSE loss"""
    target_mu = target.mean(dim=0)
    target_mu_reshaped = target_mu.unsqueeze(0).broadcast_to(target.shape)
    loss = mse(output, target) / mse(target_mu_reshaped, target)
    return loss


def get_mse_coef(activaitons):
    """Ccoefficient for MSE loss"""
    return 1 / ((activaitons.mean(dim=0) - activaitons) ** 2).mean()


def train(
    cfg: Config,
    model: PreTrainedModel,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    dataset: Union[Dataset, DatasetDict, IterableDatasetDict, IterableDataset],
    save_path: Union[str, PathLike],
):
    """Train the SAE"""

    sae = SAE(cfg)

    if isinstance(dataset, DatasetDict) or isinstance(dataset, IterableDatasetDict):
        dataset = dataset["train"]

    print("Initializing ActivationsStore")

    store = ActivationsStore(model, tokenizer, dataset, cfg)

    # Calculate MSE loss coefficient following OpenAI's approach
    sample_activations = store.next()
    mse_coef = get_mse_coef(sample_activations).item()

    optimizer = torch.optim.Adam(
        sae.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps
    )

    print("Beginning training")

    for i in tqdm.trange(cfg.total_training_batches):
        # get next batch of model activations
        x = store.next()

        # forward pass through SAE
        recons, aux_recons = sae(x)

        # calculate loss

        mse_loss = mse(recons, x)

        aux_loss = normalized_mse(
            aux_recons, x - recons.detach() + sae.b_dec.detach()
        ).nan_to_num(0)

        loss = mse_coef * mse_loss + cfg.aux_k_coef * aux_loss

        # backward pass

        loss.backward()

        sae.set_decoder_norm_to_unit_norm()
        sae.remove_gradient_parallel_to_decoder_directions()

        optimizer.step()
        optimizer.zero_grad()

        # logging

        if i % 1000 == 0:
            print(
                loss.item(),
                mse_coef * mse_loss.item(),
                cfg.aux_k_coef * aux_loss.item(),
            )

    print("Saving final model")

    sae.save(save_path)
