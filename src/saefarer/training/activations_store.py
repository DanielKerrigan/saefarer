"""
This is based on `activations_store.py` from SAELens.
"""

from typing import TYPE_CHECKING, Any, Iterator

import torch
from datasets import Dataset, IterableDataset
from einops import rearrange
from torch.utils.data import DataLoader, TensorDataset

from saefarer.constants import DTYPES

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from saefarer.training.config import TrainingConfig


class ActivationsStore:
    """
    This class is used to provide model activations from a given
    layer to train the SAE on.
    """

    def __init__(
        self,
        model: "PreTrainedModel",
        dataset: Dataset | IterableDataset | DataLoader,
        cfg: "TrainingConfig",
    ):
        self.dtype = DTYPES[cfg.dtype]
        self.device = torch.device(cfg.device)

        self.model = model.to(self.device)  # type: ignore

        self.cfg = cfg

        self._activations_dataloader: Iterator[Any] | None = None
        self._activations_storage_buffer: torch.Tensor | None = None

        if isinstance(dataset, Dataset) or isinstance(dataset, IterableDataset):
            self.dataset_dataloader = DataLoader(
                dataset,  # type: ignore
                batch_size=self.cfg.model_batch_size_sequences,
                drop_last=True,
            )
        else:
            self.dataset_dataloader = dataset

        batch_shape = next(iter(self.dataset_dataloader))[self.cfg.dataset_column].shape

        assert batch_shape[0] == self.cfg.model_batch_size_sequences, (
            f"DataLoader batch size is {batch_shape[0]} but cfg.model_batch_size_sequences = {self.cfg.model_batch_size_sequences}"
        )

        assert batch_shape[1] == self.cfg.model_sequence_length, (
            f"Dataset sequence length is {batch_shape[1]} but cfg.model_sequence_length = {self.cfg.model_sequence_length}"
        )

        self.dataset_batch_iter = iter(self.dataset_dataloader)
        self.num_samples_processed = 0

    @property
    def activations_storage_buffer(self) -> torch.Tensor:
        """
        The storage buffer contains half of the activations in the store.
        It is used to refill the dataloader when it runs out.
        """
        if self._activations_storage_buffer is None:
            self._activations_storage_buffer = self.get_buffer(
                self.cfg.n_batches_in_store // 2
            )

        return self._activations_storage_buffer

    @property
    def activations_dataloader(self) -> Iterator[Any]:
        """
        The dataloader contains half of the activations in the store
        and is iterated over to get batches of activations.
        When it runs out, more activations are retrieved and get shuffled
        with the storage buffer.
        """
        if self._activations_dataloader is None:
            self._activations_dataloader = self.get_activations_data_loader()

        return self._activations_dataloader

    @torch.no_grad()
    def get_buffer(self, n_batches, raise_at_epoch_end: bool = False) -> torch.Tensor:
        """Get buffer of activations."""
        n_tokens_in_model_batch = (
            self.cfg.model_batch_size_sequences * self.cfg.model_sequence_length
        )
        n_tokens_in_buffer = n_batches * n_tokens_in_model_batch

        new_buffer = torch.zeros(
            (n_tokens_in_buffer, self.cfg.d_in),
            dtype=self.dtype,
            requires_grad=False,
            device=self.device,
        )

        n_tokens_read = 0

        while n_tokens_read < n_tokens_in_buffer:
            tokens, attn_mask = self.get_batch_tokens(raise_at_epoch_end)
            activations = self.get_activations(tokens, attn_mask)

            n_tokens = activations.shape[0]
            # TODO: don't drop unused tokens
            n_tokens_to_use = min(n_tokens, n_tokens_in_buffer - n_tokens_read)

            start = n_tokens_read
            end = start + n_tokens_to_use
            new_buffer[start:end] = activations[0:n_tokens_to_use]

            n_tokens_read += n_tokens_to_use

        new_buffer = new_buffer[torch.randperm(new_buffer.shape[0])]

        return new_buffer

    def get_activations_data_loader(self) -> Iterator[Any]:
        """Create new dataloader."""
        try:
            new_samples = self.get_buffer(
                self.cfg.n_batches_in_store // 2, raise_at_epoch_end=True
            )
        except StopIteration as e:
            # Dump current buffer so that samples aren't leaked between epochs
            self._activations_storage_buffer = None

            print(str(e))

            try:
                new_samples = self.get_buffer(
                    self.cfg.n_batches_in_store // 2, raise_at_epoch_end=True
                )
            except StopIteration:
                raise ValueError("Unable to fill buffer after starting new epoch.")

        mixing_buffer = torch.cat([new_samples, self.activations_storage_buffer], dim=0)
        mixing_buffer = mixing_buffer[torch.randperm(mixing_buffer.shape[0])]

        self._activations_storage_buffer = mixing_buffer[: mixing_buffer.shape[0] // 2]

        dataset = TensorDataset(mixing_buffer[mixing_buffer.shape[0] // 2 :])

        # TODO: check that batch size is smaller than dataset size
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.sae_batch_size_tokens,
            shuffle=True,
        )

        dataloader_iterator = iter(dataloader)

        return dataloader_iterator

    def get_batch_tokens(
        self, raise_at_epoch_end: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Get batch of tokens from the dataset."""

        def get_tokens_and_attn_mask():
            batch = next(self.dataset_batch_iter)
            tokens = batch[self.cfg.dataset_column].to(self.device)
            mask = (
                batch[self.cfg.attn_mask_column].to(self.device)
                if self.cfg.attn_mask_column
                else None
            )
            self.num_samples_processed += self.cfg.model_batch_size_sequences
            return tokens, mask

        try:
            return get_tokens_and_attn_mask()
        except StopIteration:
            self.dataset_batch_iter = iter(self.dataset_dataloader)

            if raise_at_epoch_end:
                raise StopIteration(
                    f"Ran out of tokens in dataset after {self.num_samples_processed} samples."
                )
            else:
                return get_tokens_and_attn_mask()

    @torch.no_grad()
    def get_activations(
        self, batch_tokens: torch.Tensor, attn_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Get activations for tokens."""
        batch_output = self.model(
            batch_tokens, attention_mask=attn_mask, output_hidden_states=True
        )
        batch_activations = batch_output.hidden_states[self.cfg.hidden_state_index]
        flat_activations = rearrange(
            batch_activations, "batches seq_len d_in -> (batches seq_len) d_in"
        )

        if attn_mask is not None:
            flat_attn_mask = rearrange(
                attn_mask, "batches seq_len -> (batches seq_len)"
            ).bool()

            flat_activations = flat_activations[flat_attn_mask]

        return flat_activations

    def next_batch(self):
        """Get batch of activations."""
        try:
            return next(self.activations_dataloader)[0]
        except StopIteration:
            # if the dataloader is exhausted, create a new one
            self._activations_dataloader = self.get_activations_data_loader()
            return next(self.activations_dataloader)[0]
