"""Evaluate sparse autoencoder."""

import os
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import torch
import umap
from datasets import (
    Dataset,
    IterableDataset,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel

import saefarer.database as db
from saefarer.config import AnalysisConfig
from saefarer.model import SAE
from saefarer.types import (
    FeatureData,
    FeatureProjection,
    Histogram,
    SAEData,
    TokenSequence,
)
from saefarer.utils import (
    freedman_diaconis,
    top_k_indices,
    torch_histogram,
)


@torch.inference_mode()
def analyze(
    cfg: AnalysisConfig,
    model: PreTrainedModel,
    dataset: Union[Dataset, IterableDataset, DataLoader],
    sae: SAE,
    decode_fn: Callable[[torch.Tensor], List[str]],
    output_path: Union[str, os.PathLike],
):
    output_path = Path(output_path)

    if output_path.exists():
        raise OSError(f"{output_path} already exists")

    model.to(cfg.device)  # type: ignore

    ds = _get_dataset(dataset, cfg)

    con, cur = db.create_database(output_path)

    # this is in preparation of supporting multiple SAEs
    sae_id = "default"

    feature_indices = cfg.feature_indices or list(range(sae.cfg.d_sae))

    dead_feature_ids, alive_feature_ids = _get_dead_alive_features(sae, feature_indices)
    num_alive_features = len(alive_feature_ids)
    num_dead_features = len(dead_feature_ids)

    feature_batches = [
        alive_feature_ids[i : i + cfg.feature_batch_size]
        for i in range(0, num_alive_features, cfg.feature_batch_size)
    ]

    firing_rates = []

    features_processed = 0
    progress_bar = tqdm(total=num_alive_features, desc="Calculating feature data")

    for features in feature_batches:
        sae_activations = _get_sae_activations(features, sae, model, ds, cfg)

        for i, feature in enumerate(features):
            feature_activations = sae_activations[..., i]

            feature_data = _get_feature_data(
                feature, sae_id, feature_activations, decode_fn, ds, cfg
            )
            firing_rates.append(feature_data["firing_rate"])

            db.insert_feature(feature_data, con, cur)

            features_processed += 1
            progress_bar.update(features_processed)

    progress_bar.close()

    firing_rate_histogram = _get_firing_rate_histogram(firing_rates)

    feature_projection = _get_feature_projection(sae, alive_feature_ids)

    sae_data = SAEData(
        sae_id=sae_id,
        num_alive_features=num_alive_features,
        num_dead_features=num_dead_features,
        alive_feature_ids=alive_feature_ids,
        dead_feature_ids=dead_feature_ids,
        firing_rate_histogram=firing_rate_histogram,
        feature_projection=feature_projection,
    )

    db.insert_sae(sae_data, con, cur)


@torch.inference_mode()
def _get_sae_activations(
    feature_indices: List[int],
    sae: SAE,
    model: PreTrainedModel,
    ds: Dict[str, torch.Tensor],
    cfg: AnalysisConfig,
) -> torch.Tensor:
    sae_activations = []

    token_batches = ds[cfg.dataset_column].split(cfg.model_batch_size_sequences)

    for token_batch in token_batches:
        token_batch = token_batch.to(cfg.device)
        batch_model_output = model(token_batch, output_hidden_states=True)
        batch_model_acts = batch_model_output.hidden_states[sae.cfg.hidden_state_index]
        batch_sae_acts, _ = sae.encode(batch_model_acts)
        sae_activations.append(batch_sae_acts[..., feature_indices])

    sae_activations = torch.cat(sae_activations, dim=0)

    return sae_activations


def _get_feature_data(
    feature_id: int,
    sae_id: str,
    feature_activations: torch.Tensor,
    decode_fn: Callable[[torch.Tensor], List[str]],
    ds: Dict[str, torch.Tensor],
    cfg: AnalysisConfig,
) -> FeatureData:
    sequences = _get_sequence_data(decode_fn, ds, feature_activations, cfg)

    positive_activations = feature_activations[feature_activations > 0]
    firing_rate = positive_activations.numel() / feature_activations.numel()

    activations_histogram = _get_activation_histogram(positive_activations)

    return FeatureData(
        feature_id=feature_id,
        sae_id=sae_id,
        firing_rate=firing_rate,
        activations_histogram=activations_histogram,
        sequences=sequences,
    )


@torch.inference_mode()
def _get_sequence_data(
    decode_fn: Callable[[torch.Tensor], List[str]],
    ds: Dict[str, torch.Tensor],
    feature_activations: torch.Tensor,
    cfg: AnalysisConfig,
) -> List[TokenSequence]:
    top_indices = top_k_indices(
        feature_activations, k=cfg.n_example_sequences, largest=True
    )

    sequences: List[TokenSequence] = []

    for point in top_indices:
        seq_i = int(point[0].item())
        tok_i = int(point[1].item())

        min_tok_i = max(0, tok_i - cfg.n_context_tokens)
        max_tok_i = min(cfg.model_sequence_length, tok_i + cfg.n_context_tokens)

        tok_ids = ds[cfg.dataset_column][seq_i, min_tok_i : max_tok_i + 1]
        acts = feature_activations[seq_i, min_tok_i : max_tok_i + 1]

        token_sequence = TokenSequence(
            token=decode_fn(tok_ids),
            activation=acts.tolist(),
            max_index=tok_i - min_tok_i,
        )
        sequences.append(token_sequence)

    return sequences


@torch.inference_mode()
def _get_activation_histogram(
    positive_activations: torch.Tensor,
) -> Histogram:
    num_bins = min(freedman_diaconis(positive_activations), 100)
    counts, thresholds = torch_histogram(positive_activations, bins=num_bins)
    return Histogram(counts=counts.tolist(), thresholds=thresholds.tolist())


@torch.inference_mode()
def _get_firing_rate_histogram(
    firing_rates: List[float],
) -> Histogram:
    log_rates = np.log10(firing_rates)
    counts, thresholds = np.histogram(log_rates, bins="fd")
    return Histogram(counts=counts.tolist(), thresholds=thresholds.tolist())


@torch.inference_mode()
def _get_dead_alive_features(
    sae: SAE, feature_indices: List[int]
) -> Tuple[List[int], List[int]]:
    dead_mask = sae.get_dead_neuron_mask()

    dead_features = torch.nonzero(dead_mask, as_tuple=True)[0].tolist()
    alive_features = torch.nonzero(~dead_mask, as_tuple=True)[0].tolist()

    user_indices = set(feature_indices)

    dead_features = [i for i in dead_features if i in user_indices]
    alive_features = [i for i in alive_features if i in user_indices]

    return dead_features, alive_features


@torch.inference_mode()
def _get_feature_projection(sae: SAE, feature_ids: List[int]) -> FeatureProjection:
    n_features = len(feature_ids)

    # UMAP doesn't work with <= 2 datapoints, so in these cases we will
    # return a fake projection.
    if n_features <= 2:
        pos = [float(x) for x in range(n_features)]
        return FeatureProjection(feature_id=feature_ids, x=pos, y=pos)

    weights: np.ndarray = sae.W_dec.numpy(force=True)

    n_neighbors = min(len(feature_ids) - 1, 15)

    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=2)
    weights_embedded: np.ndarray = reducer.fit_transform(weights[feature_ids])  # type: ignore

    x: List[float] = weights_embedded[:, 0].tolist()
    y: List[float] = weights_embedded[:, 1].tolist()

    return FeatureProjection(feature_id=feature_ids, x=x, y=y)


def _get_dataset(
    dataset: Union[Dataset, IterableDataset, DataLoader], cfg: AnalysisConfig
) -> Dict[str, torch.Tensor]:
    if isinstance(dataset, Dataset):
        return dataset[0 : cfg.total_analysis_sequences]

    if isinstance(dataset, IterableDataset):
        dataloader = DataLoader(
            dataset,  # type: ignore
            batch_size=cfg.total_analysis_sequences,
        )
    else:
        dataloader = dataset

    original_batch_size = dataloader.batch_size
    dataloader.batch_size = cfg.total_analysis_sequences

    ds = next(iter(dataloader))

    dataloader.batch_size = original_batch_size

    return ds
