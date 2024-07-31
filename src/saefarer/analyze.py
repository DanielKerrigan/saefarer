"""Evaluate sparse autoencoder."""

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import umap
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from saefarer.config import Config
from saefarer.feature_data import (
    FeatureData,
    FeatureProjection,
    Histogram,
    SAEData,
    TokenSequence,
)
from saefarer.model import SAE
from saefarer.utils import freedman_diaconis, top_k_indices


@torch.inference_mode()
def analyze_sae(
    sae: SAE,
    model: PreTrainedModel,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    tokens: torch.Tensor,
    feature_indices: List[int],
    cfg: Config,
    output_dir: Union[str, os.PathLike],
):
    sae_activations = _get_sae_activations(sae, model, tokens, cfg)

    output_dir = Path(output_dir)

    if not output_dir.exists():
        raise OSError(f"{output_dir} does not exist")

    dead_indices, alive_indices = _get_dead_alive_features(sae, feature_indices)

    firing_rates = []

    num_digits = math.floor(math.log10(max([1, *alive_indices]))) + 1
    feature_index_to_path: Dict[int, str] = {}

    overview_path = output_dir / "overview.json"
    features_dir = output_dir / "features"
    features_dir.mkdir(exist_ok=True)

    for i in tqdm(alive_indices, desc="Computing data for feature"):
        feature_activations = sae_activations[..., i]

        sequences = _get_sequence_data(tokenizer, tokens, feature_activations, cfg)

        positive_activations = feature_activations[feature_activations > 0]
        firing_rate = positive_activations.numel() / feature_activations.numel()

        firing_rates.append(firing_rate)

        activations_histogram = _get_activation_histogram(positive_activations)

        feature_data = FeatureData(
            firing_rate=firing_rate,
            activations_histogram=activations_histogram,
            sequences=sequences,
        )

        feature_path = features_dir / f"{i:0{num_digits}}.json"
        feature_index_to_path[i] = feature_path.relative_to(output_dir).as_posix()
        feature_path.write_text(json.dumps(feature_data), encoding="utf-8")

    firing_rate_histogram = _get_firing_rate_histogram(firing_rates)

    feature_projection = _get_feature_projection(sae, feature_indices)

    sae_data = SAEData(
        num_dead_features=len(dead_indices),
        num_alive_features=len(alive_indices),
        firing_rate_histogram=firing_rate_histogram,
        feature_projection=feature_projection,
        dead_feature_indices=dead_indices,
        alive_feature_indices=alive_indices,
        feature_index_to_path=feature_index_to_path,
    )

    overview_path.write_text(json.dumps(sae_data), encoding="utf-8")


@torch.inference_mode()
def _get_sae_activations(
    sae: SAE, model: PreTrainedModel, tokens: torch.Tensor, cfg: Config
) -> torch.Tensor:
    sae_activations = []

    token_batches = tokens.split(cfg.model_batch_size_sequences)

    for token_batch in tqdm(token_batches, desc="Computing SAE activations"):
        batch_model_output = model(token_batch, output_hidden_states=True)
        batch_model_acts = batch_model_output.hidden_states[cfg.hidden_state_index]
        batch_sae_acts, _ = sae.encode(batch_model_acts)
        sae_activations.append(batch_sae_acts)

    sae_activations = torch.cat(sae_activations, dim=0)

    return sae_activations


@torch.inference_mode()
def _get_sequence_data(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    tokens: torch.Tensor,
    feature_activations: torch.Tensor,
    cfg: Config,
) -> List[TokenSequence]:
    top_indices = top_k_indices(feature_activations, k=10, largest=True)

    sequences: List[TokenSequence] = []

    for point in top_indices:
        seq_i = int(point[0].item())
        tok_i = int(point[1].item())

        min_tok_i = max(0, tok_i - 5)
        max_tok_i = min(cfg.model_sequence_length, tok_i + 5)

        tok_ids = tokens[seq_i, min_tok_i : max_tok_i + 1]
        acts = feature_activations[seq_i, min_tok_i : max_tok_i + 1]

        token_sequence = TokenSequence(
            token=tokenizer.batch_decode(tok_ids),
            activation=acts.tolist(),
            max_index=tok_i - min_tok_i,
        )
        sequences.append(token_sequence)

    return sequences


@torch.inference_mode()
def _get_activation_histogram(
    positive_activations: torch.Tensor,
) -> Histogram:
    num_bins = freedman_diaconis(positive_activations)
    counts, thresholds = torch.histogram(positive_activations, bins=num_bins)
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
def _get_feature_projection(sae: SAE, feature_indices: List[int]) -> FeatureProjection:
    weights = sae.W_dec.numpy()

    reducer = umap.UMAP()
    weights_embedded: np.ndarray = reducer.fit_transform(weights)  # type: ignore

    x = weights_embedded[:, 0].tolist()
    y = weights_embedded[:, 1].tolist()

    return FeatureProjection(feature_index=feature_indices, x=x, y=y)
