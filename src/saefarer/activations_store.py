""""""

from typing import Any, Iterator, Union

import torch
from datasets import Dataset, IterableDataset
from einops import rearrange
from torch.utils.data import DataLoader, TensorDataset
from transformers import PreTrainedModel

from saefarer.config import Config
from saefarer.constants import DTYPES


class ActivationsStore:
    """
    This class is used to provide model activations from a given
    layer to train the SAE on.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        dataset: Union[Dataset, IterableDataset, DataLoader],
        cfg: Config,
    ):
        self.dtype = DTYPES[cfg.dtype]
        self.device = torch.device(cfg.device)

        self.model = model

        # self.dataset = dataset
        self.cfg = cfg

        self._activations_dataloader: Union[Iterator[Any], None] = None
        self._activations_storage_buffer: Union[torch.Tensor, None] = None

        if isinstance(dataset, Dataset) or isinstance(dataset, IterableDataset):
            self.dataset_dataloader = DataLoader(
                dataset,  # type: ignore
                batch_size=self.cfg.model_batch_size_sequences,
            )
        else:
            self.dataset_dataloader = dataset

        batch_shape = next(iter(self.dataset_dataloader))[self.cfg.dataset_column].shape

        assert (
            batch_shape[0] == self.cfg.model_batch_size_sequences
        ), f"DataLoader batch size is {batch_shape[0]} but cfg.model_batch_size_sequences = {self.cfg.model_batch_size_sequences}"

        assert (
            batch_shape[1] == self.cfg.model_sequence_length
        ), f"Dataset sequence length is {batch_shape[1]} but cfg.model_sequence_length = {self.cfg.model_sequence_length}"

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
        When it runs out, more activations are retrived and get shuffled
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

        new_buffer = torch.zeros(
            (n_batches * n_tokens_in_model_batch, self.cfg.d_in),
            dtype=self.dtype,
            requires_grad=False,
            device=self.device,
        )

        for i in range(n_batches):
            tokens = self.get_batch_tokens(raise_at_epoch_end)
            activations = self.get_activations(tokens)

            start = i * n_tokens_in_model_batch
            end = start + n_tokens_in_model_batch
            new_buffer[start:end] = activations

        new_buffer = new_buffer[torch.randperm(new_buffer.shape[0])]

        return new_buffer

    def get_activations_data_loader(self) -> Iterator[Any]:
        """Create new dataloader."""
        try:
            new_samples = self.get_buffer(
                self.cfg.n_batches_in_store // 2, raise_at_epoch_end=True
            )
        except StopIteration as e:
            print(e.value)
            # Dump current buffer so that samples aren't leaked between epochs
            self._activations_storage_buffer = None

            try:
                new_samples = self.get_buffer(
                    self.cfg.n_batches_in_store // 2, raise_at_epoch_end=True
                )
            except StopIteration as e:
                raise ValueError("Unable to fill buffer after starting new epoch.")

        mixing_buffer = torch.cat([new_samples, self.activations_storage_buffer], dim=0)
        mixing_buffer = mixing_buffer[torch.randperm(mixing_buffer.shape[0])]

        self._activations_storage_buffer = mixing_buffer[: mixing_buffer.shape[0] // 2]

        dataset = TensorDataset(mixing_buffer[mixing_buffer.shape[0] // 2 :])

        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.sae_batch_size_tokens,
            shuffle=True,
        )

        dataloader_iterator = iter(dataloader)

        return dataloader_iterator

    def get_batch_tokens(self, raise_at_epoch_end: bool = False) -> torch.Tensor:
        """Get batch of tokens from the dataset."""
        try:
            batch = next(self.dataset_batch_iter)[self.cfg.dataset_column]
            self.num_samples_processed += self.cfg.model_batch_size_sequences
            return batch.to(self.device)
        except StopIteration:
            self.dataset_batch_iter = iter(self.dataset_dataloader)

            if raise_at_epoch_end:
                raise StopIteration(
                    f"Ran out of tokens in dataset after {self.num_samples_processed} samples."
                )
            else:
                batch = next(self.dataset_batch_iter)[self.cfg.dataset_column]
                self.num_samples_processed += self.cfg.model_batch_size_sequences
                return batch.to(self.device)

    @torch.no_grad()
    def get_activations(self, batch_tokens: torch.Tensor) -> torch.Tensor:
        """Get activations for tokens."""
        batch_output = self.model(batch_tokens, output_hidden_states=True)
        batch_activations = batch_output.hidden_states[self.cfg.hidden_state_index]
        flat_activations = rearrange(
            batch_activations, "batches seq_len d_in -> (batches seq_len) d_in"
        )
        return flat_activations

    def next_batch(self):
        """Get batch of activations."""
        try:
            return next(self.activations_dataloader)[0]
        except StopIteration:
            # if the dataloader is exhausted, create a new one
            self._activations_dataloader = self.get_activations_data_loader()
            return next(self.activations_dataloader)[0]
