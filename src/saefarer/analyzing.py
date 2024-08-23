"""Analyze sparse autoencoder."""

import math
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
    CumSumPercentL1Norm,
    CumSumPercentL1NormRange,
    FeatureData,
    FeatureProjection,
    FeatureTokenSequence,
    Histogram,
    SAEData,
)
from saefarer.utils import (
    freedman_diaconis_np,
    freedman_diaconis_torch,
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

    activation_rates = []
    n_neurons_majority_l1_norm = []

    progress_bar = tqdm(
        total=num_alive_features,
        desc="Calculating feature data",
        disable=not cfg.show_progress,
    )

    min_cumsum_percent_l1_norm: torch.Tensor = torch.empty(0)
    max_cumsum_percent_l1_norm: torch.Tensor = torch.empty(0)

    for features in feature_batches:
        sae_activations = _get_sae_activations(features, sae, model, ds, cfg)

        for i, feature in enumerate(features):
            feature_activations = sae_activations[..., i]

            feature_data = _get_feature_data(
                feature, sae_id, sae, feature_activations, decode_fn, ds, cfg
            )

            cumsum: torch.Tensor = torch.Tensor(
                feature_data["cumsum_percent_l1_norm"]["cum_sum"]
            )

            if min_cumsum_percent_l1_norm.numel() == 0:
                min_cumsum_percent_l1_norm = cumsum
                max_cumsum_percent_l1_norm = cumsum
            else:
                min_cumsum_percent_l1_norm = torch.min(
                    input=torch.stack((min_cumsum_percent_l1_norm, cumsum), dim=0),
                    dim=0,
                ).values
                max_cumsum_percent_l1_norm = torch.max(
                    torch.stack((max_cumsum_percent_l1_norm, cumsum), dim=0), dim=0
                ).values

            activation_rates.append(feature_data["activation_rate"])
            n_neurons_majority_l1_norm.append(
                feature_data["n_neurons_majority_l1_norm"]
            )

            db.insert_feature(feature_data, con, cur)

            progress_bar.update()

    progress_bar.close()

    activation_rate_histogram = _get_activation_rate_histogram(activation_rates)

    feature_projection = _get_feature_projection(sae, alive_feature_ids)

    dimensionality_histogram = _get_dimensionality_histogram(n_neurons_majority_l1_norm)

    cumsum_percent_l1_norm_range = CumSumPercentL1NormRange(
        mins=min_cumsum_percent_l1_norm.tolist(),
        maxs=max_cumsum_percent_l1_norm.tolist(),
    )

    sae_data = SAEData(
        sae_id=sae_id,
        num_alive_features=num_alive_features,
        num_dead_features=num_dead_features,
        alive_feature_ids=alive_feature_ids,
        dead_feature_ids=dead_feature_ids,
        activation_rate_histogram=activation_rate_histogram,
        dimensionality_histogram=dimensionality_histogram,
        cumsum_percent_l1_norm_range=cumsum_percent_l1_norm_range,
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
    tokens = ds[cfg.dataset_column]

    sae_activations = torch.zeros(
        tokens.shape + (len(feature_indices),), device=cfg.device, dtype=sae.dtype
    )
    offset = 0

    token_batches = tokens.split(cfg.model_batch_size_sequences)

    for token_batch in token_batches:
        token_batch = token_batch.to(cfg.device)
        batch_model_output = model(token_batch, output_hidden_states=True)
        batch_model_acts = batch_model_output.hidden_states[sae.cfg.hidden_state_index]
        batch_sae_acts, _ = sae.encode(batch_model_acts)

        start = offset
        offset += batch_sae_acts.shape[0]
        end = offset

        sae_activations[start:end, :, :] = batch_sae_acts[..., feature_indices]

    return sae_activations


@torch.inference_mode()
def _get_feature_data(
    feature_id: int,
    sae_id: str,
    sae: SAE,
    feature_activations: torch.Tensor,
    decode_fn: Callable[[torch.Tensor], List[str]],
    ds: Dict[str, torch.Tensor],
    cfg: AnalysisConfig,
) -> FeatureData:
    sequences = _get_sequence_data(decode_fn, ds, feature_activations, cfg)

    positive_activations = feature_activations[feature_activations > 0]
    activation_rate = positive_activations.numel() / feature_activations.numel()

    activations_histogram = _get_activation_histogram(positive_activations)
    cumsum_percent_l1_norm, n_neurons_majority_l1_norm = _get_cumsum_percent_l1_norm(
        sae.W_dec[feature_id]
    )

    return FeatureData(
        sae_id=sae_id,
        feature_id=feature_id,
        activation_rate=activation_rate,
        max_activation=feature_activations.max().item(),
        n_neurons_majority_l1_norm=n_neurons_majority_l1_norm,
        cumsum_percent_l1_norm=cumsum_percent_l1_norm,
        activations_histogram=activations_histogram,
        sequences=sequences,
    )


@torch.inference_mode()
def _get_sequence_data(
    decode_fn: Callable[[torch.Tensor], List[str]],
    ds: Dict[str, torch.Tensor],
    feature_activations: torch.Tensor,
    cfg: AnalysisConfig,
) -> Dict[str, List[FeatureTokenSequence]]:
    sequence_indices: Dict[str, torch.Tensor] = {}

    top_indices = top_k_indices(
        feature_activations, k=cfg.n_example_sequences, largest=True
    )

    sequence_indices["Max Activations"] = top_indices

    max_act = feature_activations.max()
    activation_ranges = torch.linspace(0, max_act, cfg.n_sequence_intervals + 1)
    interval_min_max = reversed(list(zip(activation_ranges, activation_ranges[1:])))

    for i, (interval_min, interval_max) in enumerate(interval_min_max):
        valid_indices = torch.stack(
            torch.where(
                (feature_activations >= interval_min)
                & (feature_activations < interval_max)
            ),
            dim=-1,
        )

        if valid_indices.shape[0] > cfg.n_example_sequences:
            rand_indices = torch.multinomial(
                input=torch.ones(valid_indices.shape[0]),
                num_samples=cfg.n_example_sequences,
                replacement=False,
            )
            valid_indices = valid_indices[rand_indices]

        sequence_indices[f"Interval {i + 1}"] = valid_indices

    sequences: Dict[str, List[FeatureTokenSequence]] = {}

    for key, indices in sequence_indices.items():
        key_seq: List[FeatureTokenSequence] = []

        for point in indices:
            seq_i = int(point[0].item())
            tok_i = int(point[1].item())

            min_tok_i = max(0, tok_i - cfg.n_context_tokens)
            max_tok_i = min(cfg.model_sequence_length, tok_i + cfg.n_context_tokens)

            tok_ids = ds[cfg.dataset_column][seq_i, min_tok_i : max_tok_i + 1]
            acts = feature_activations[seq_i, min_tok_i : max_tok_i + 1]

            extras: Dict[str, List[str]] = {}

            for entry in cfg.extra_token_columns:
                if isinstance(entry, str):
                    col, fmt = entry, str
                else:
                    col, fmt = entry

                values = ds[col][seq_i]

                if values.dim() == 0:
                    values = [values.item()] * tok_ids.shape[0]
                else:
                    values = values[min_tok_i : max_tok_i + 1].tolist()

                extras[col] = [fmt(value) for value in values]

            token_sequence = FeatureTokenSequence(
                token=decode_fn(tok_ids),
                activation=acts.tolist(),
                extras=extras,
                max_index=tok_i - min_tok_i,
            )
            key_seq.append(token_sequence)

        sequences[key] = key_seq

    return sequences


@torch.inference_mode()
def _get_activation_histogram(
    positive_activations: torch.Tensor,
) -> Histogram:
    num_bins = min(freedman_diaconis_torch(positive_activations), 64)
    counts, thresholds = torch_histogram(positive_activations, bins=num_bins)
    return Histogram(counts=counts.tolist(), thresholds=thresholds.tolist())


@torch.inference_mode()
def _get_activation_rate_histogram(
    activation_rates: List[float],
) -> Histogram:
    log_rates = np.log10(activation_rates)
    counts, thresholds = np.histogram(log_rates, bins="fd")
    return Histogram(counts=counts.tolist(), thresholds=thresholds.tolist())


@torch.inference_mode()
def _get_dimensionality_histogram(n_neurons_majority_l1_norm: List[int]) -> Histogram:
    array = np.array(n_neurons_majority_l1_norm)
    num_bins = min(freedman_diaconis_np(array), 64)
    counts, thresholds = np.histogram(array, bins=num_bins)
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


@torch.inference_mode()
def _get_cumsum_percent_l1_norm(
    weights: torch.Tensor,
) -> Tuple[CumSumPercentL1Norm, int]:
    d_in = weights.shape[0]
    cum_sum_abs_weights = torch.cumsum(
        weights.abs().sort(descending=True).values, dim=0
    )
    # imprecision can lead to value slightly above 1
    cumsum_percent_l1_norm = torch.clamp(cum_sum_abs_weights / weights.norm(p=1), max=1)
    step_size = math.ceil(weights.shape[0] / 64)
    indices = torch.arange(0, d_in, step_size)

    n_neurons_majority_l1_norm = int(
        torch.where(cumsum_percent_l1_norm >= 0.5)[0][0] + 1
    )

    return (
        CumSumPercentL1Norm(
            n_neurons=(indices + 1).tolist(),
            cum_sum=cumsum_percent_l1_norm[indices].tolist(),
        ),
        n_neurons_majority_l1_norm,
    )


@torch.inference_mode()
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
        dataloader = DataLoader(
            dataset=dataset.dataset,
            shuffle=False,
            batch_size=cfg.total_analysis_sequences,
            collate_fn=dataset.collate_fn,
            num_workers=dataset.num_workers,
        )

    ds = next(iter(dataloader))

    return ds
