"""
This is based on the SAE analysis code from sae_vis and SAEDashboard:
https://github.com/callummcdougall/sae_vis
https://github.com/jbloomAus/SAEDashboard
"""

import dataclasses
import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import umap
from datasets import (
    Dataset,
    IterableDataset,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

import saefarer.analysis.database as db
from saefarer import sae
from saefarer.analysis.feature_analysis import get_feature_data
from saefarer.analysis.model_and_dataset import (
    get_dataset_info,
    get_dataset_with_predictions,
    get_model_info,
)
from saefarer.analysis.types import (
    FeatureProjection,
    HistogramData,
    SAEData,
)

if TYPE_CHECKING:
    from os import PathLike

    from transformers import PreTrainedModel, PreTrainedTokenizer

    from saefarer.analysis.config import AnalysisConfig


@torch.inference_mode()
def analyze(
    cfg: "AnalysisConfig",
    model: "PreTrainedModel",
    dataset: Dataset | IterableDataset | DataLoader,
    sae: "sae.SAE",
    tokenizer: "PreTrainedTokenizer",
    output_path: "str | PathLike",
):
    output_path = Path(output_path)

    if output_path.exists():
        raise OSError(f"{output_path} already exists")

    con, cur = db.create_database(output_path)

    model.to(cfg.device)  # type: ignore

    rng = np.random.default_rng()

    ds = get_dataset_with_predictions(model, dataset, cfg)
    dataset_info = get_dataset_info(cfg)
    model_info = get_model_info(ds, dataset_info)

    db.insert_misc("analysis_cfg", dataclasses.asdict(cfg), con, cur)
    db.insert_misc("dataset_info", dataset_info, con, cur)
    db.insert_misc("model_info", model_info, con, cur)

    # this is in preparation of supporting multiple SAEs
    sae_id = "default"

    feature_indices = cfg.feature_indices or list(range(sae.cfg.d_sae))

    dead_feature_ids, alive_feature_ids = _get_dead_alive_features(sae, feature_indices)
    n_alive_features = len(alive_feature_ids)
    n_dead_features = len(dead_feature_ids)

    feature_batches = [
        alive_feature_ids[i : i + cfg.feature_batch_size]
        for i in range(0, n_alive_features, cfg.feature_batch_size)
    ]

    print("Beginning to analyze features", datetime.datetime.now())

    progress_bar = tqdm(
        total=n_alive_features,
        desc="Calculating feature data",
        disable=not cfg.show_progress,
    )

    non_activating_feature_ids = []
    token_act_rates = []
    sequence_act_rates = []

    for features in feature_batches:
        sae_activations = _get_sae_activations(features, sae, model, ds, cfg)

        for i, feature in enumerate(features):
            feature_activations = sae_activations[..., i]

            positive_activation_mask = feature_activations > 0

            if positive_activation_mask.sum() == 0:
                non_activating_feature_ids.append(feature)
            else:
                feature_data = get_feature_data(
                    feature,
                    sae_id,
                    dataset_info,
                    model_info,
                    feature_activations,
                    positive_activation_mask,
                    tokenizer,
                    ds,
                    cfg,
                    rng,
                )

                token_act_rates.append(feature_data["token_act_rate"])
                sequence_act_rates.append(feature_data["sequence_act_rate"])

                db.insert_feature(feature_data, con, cur)

            progress_bar.update()

    progress_bar.close()

    print("Finished analyzing features", datetime.datetime.now())

    if non_activating_feature_ids:
        alive_feature_ids = list(
            set(alive_feature_ids) - set(non_activating_feature_ids)
        )
        n_alive_features = len(alive_feature_ids)

    n_non_activating_features = len(non_activating_feature_ids)

    token_act_rate_histogram = _get_activation_rate_histogram(token_act_rates)
    sequence_act_rate_histogram = _get_activation_rate_histogram(sequence_act_rates)

    feature_projection = _get_feature_projection(sae, alive_feature_ids)

    sae_data = SAEData(
        sae_id=sae_id,
        n_total_features=len(feature_indices),
        n_alive_features=n_alive_features,
        n_dead_features=n_dead_features,
        n_non_activating_features=n_non_activating_features,
        alive_feature_ids=alive_feature_ids,
        token_act_rate_histogram=token_act_rate_histogram,
        sequence_act_rate_histogram=sequence_act_rate_histogram,
        feature_projection=feature_projection,
    )

    db.insert_sae(sae_data, con, cur)


@torch.inference_mode()
def _get_sae_activations(
    feature_indices: list[int],
    sae: "sae.SAE",
    model: "PreTrainedModel",
    ds: dict[str, torch.Tensor],
    cfg: "AnalysisConfig",
) -> torch.Tensor:
    tokens = ds[cfg.tokens_column]
    attn_masks = ds[cfg.attn_mask_column]

    sae_activations = torch.zeros(
        tokens.shape + (len(feature_indices),), device=cfg.device, dtype=sae.dtype
    )
    offset = 0

    token_batches = tokens.split(cfg.model_batch_size_sequences)
    attn_mask_batches = attn_masks.split(cfg.model_batch_size_sequences)
    # TODO: consider caching these activations so that they don't have to be
    # re-computed for each batch of features.
    for token_batch, attn_mask_batch in zip(token_batches, attn_mask_batches):
        token_batch = token_batch.to(cfg.device)
        attn_mask_batch = attn_mask_batch.to(cfg.device)
        batch_model_output = model(
            token_batch,
            attention_mask=attn_mask_batch,
            output_hidden_states=True,
        )
        batch_model_acts = batch_model_output.hidden_states[sae.cfg.hidden_state_index]
        batch_sae_acts, _ = sae.encode(batch_model_acts)

        start = offset
        offset += batch_sae_acts.shape[0]
        end = offset

        sae_activations[start:end, :, :] = batch_sae_acts[..., feature_indices]

    return sae_activations


@torch.inference_mode()
def _get_activation_rate_histogram(activation_rates: list[float]) -> HistogramData:
    log_rates = np.log10(activation_rates)
    counts, thresholds = np.histogram(log_rates, "fd")
    return HistogramData(counts=counts.tolist(), thresholds=thresholds.tolist())


@torch.inference_mode()
def _get_dead_alive_features(
    sae: "sae.SAE", feature_indices: list[int]
) -> tuple[list[int], list[int]]:
    dead_mask = sae.get_dead_neuron_mask()

    dead_features = torch.nonzero(dead_mask, as_tuple=True)[0].tolist()
    alive_features = torch.nonzero(~dead_mask, as_tuple=True)[0].tolist()

    user_indices = set(feature_indices)

    dead_features = [i for i in dead_features if i in user_indices]
    alive_features = [i for i in alive_features if i in user_indices]

    return dead_features, alive_features


@torch.inference_mode()
def _get_feature_projection(
    sae: "sae.SAE", feature_ids: list[int]
) -> FeatureProjection:
    n_features = len(feature_ids)

    # UMAP doesn't work with <= 2 datapoints, so in these cases we will
    # return a fake projection.
    if n_features <= 2:
        pos = [float(x) for x in range(n_features)]
        return FeatureProjection(feature_ids=feature_ids, xs=pos, ys=pos)

    weights: np.ndarray = sae.W_dec.numpy(force=True)

    n_neighbors = min(len(feature_ids) - 1, 15)

    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=2)
    weights_embedded: np.ndarray = reducer.fit_transform(weights[feature_ids])  # type: ignore

    x: list[float] = weights_embedded[:, 0].tolist()
    y: list[float] = weights_embedded[:, 1].tolist()

    return FeatureProjection(feature_ids=feature_ids, xs=x, ys=y)
