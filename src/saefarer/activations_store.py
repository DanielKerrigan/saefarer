""""""

from typing import Any, Iterator, Optional, Union

import torch
from datasets import Dataset, IterableDataset
from einops import rearrange
from torch.utils.data import DataLoader, TensorDataset
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from saefarer.config import Config
from saefarer.constants import DTYPES
from saefarer.tokenize_and_concat import tokenize_and_concat_iterator


class ActivationsStore:
    """
    This class is used to provide model activations from a given
    layer to train the SAE on.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]],
        dataset: Union[Dataset, IterableDataset],
        cfg: Config,
    ):
        self.dtype = DTYPES[cfg.dtype]
        self.device = torch.device(cfg.device)

        self.model = model

        self.tokenizer = tokenizer
        self.dataset = dataset
        self.cfg = cfg

        self._dataloader: Union[Iterator[Any], None] = None
        self._storage_buffer: Union[torch.Tensor, None] = None

        if self.cfg.is_dataset_tokenized:
            seq_length = next(iter(dataset))[self.cfg.dataset_column].shape[0]
            assert (
                seq_length == self.cfg.model_sequence_length
            ), f"Dataset sequence length is {seq_length} but cfg.model_sequence_length = {self.cfg.model_sequence_length}"

        self.tokenized_sequences = self._get_tokenized_sequences_iterator()
        self.num_samples_processed = 0

    @property
    def storage_buffer(self) -> torch.Tensor:
        """
        The storage buffer contains half of the activations in the store.
        It is used to refill the dataloader when it runs out.
        """
        if self._storage_buffer is None:
            self._storage_buffer = self.get_buffer(self.cfg.n_batches_in_store // 2)

        return self._storage_buffer

    @property
    def dataloader(self) -> Iterator[Any]:
        """
        The dataloader contains half of the activations in the store
        and is iterated over to get batches of activations.
        When it runs out, more activations are retrived and get shuffled
        with the storage buffer.
        """
        if self._dataloader is None:
            self._dataloader = self.get_data_loader()

        return self._dataloader

    def _iterate_raw_dataset(self) -> Iterator[Union[torch.Tensor, list[int], str]]:
        """Iterator over the rows of the dataset."""
        for row in self.dataset:
            yield row[self.cfg.dataset_column]  # type: ignore
            self.num_samples_processed += 1

    def _get_tokenized_sequences_iterator(self) -> Iterator[torch.Tensor]:
        """Iterator over tokenized sequences."""

        # if the dataset is tokenized, just iterate over the rows
        if self.cfg.is_dataset_tokenized:
            for row in self._iterate_raw_dataset():
                if isinstance(row, torch.Tensor):
                    yield row
                else:
                    yield torch.tensor(
                        row, dtype=torch.long, device=self.device, requires_grad=False
                    )
        else:
            # if the dataset is not tokenized, tokenize it on the fly
            assert self.tokenizer is not None
            bos_token_id = (
                self.tokenizer.bos_token_id if self.cfg.prepend_bos_token else None
            )
            yield from tokenize_and_concat_iterator(
                dataset_iterator=self._iterate_raw_dataset(),  # type: ignore
                tokenizer=self.tokenizer,
                context_size=self.cfg.model_sequence_length,
                begin_batch_token_id=bos_token_id,
                begin_sequence_token_id=None,
                sequence_separator_token_id=bos_token_id,
            )

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

    def get_data_loader(self) -> Iterator[Any]:
        """Create new dataloader."""
        try:
            new_samples = self.get_buffer(
                self.cfg.n_batches_in_store // 2, raise_at_epoch_end=True
            )
        except StopIteration as e:
            print(e.value)
            # Dump current buffer so that samples aren't leaked between epochs
            self._storage_buffer = None

            try:
                new_samples = self.get_buffer(
                    self.cfg.n_batches_in_store // 2, raise_at_epoch_end=True
                )
            except StopIteration as e:
                raise ValueError("Unable to fill buffer after starting new epoch.")

        mixing_buffer = torch.cat([new_samples, self.storage_buffer], dim=0)
        mixing_buffer = mixing_buffer[torch.randperm(mixing_buffer.shape[0])]

        self._storage_buffer = mixing_buffer[: mixing_buffer.shape[0] // 2]

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

        batch_sequences = []

        for _ in range(self.cfg.model_batch_size_sequences):
            try:
                batch_sequences.append(next(self.tokenized_sequences))
            except StopIteration:
                self.tokenized_sequences = self._get_tokenized_sequences_iterator()

                if raise_at_epoch_end:
                    raise StopIteration(
                        f"Ran out of tokens in dataset after {self.num_samples_processed} samples."
                    )
                else:
                    batch_sequences.append(next(self.tokenized_sequences))

        batch = torch.stack(batch_sequences, dim=0).to(self.device)

        return batch

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
            return next(self.dataloader)[0]
        except StopIteration:
            # if the dataloader is exhausted, create a new one
            self._dataloader = self.get_data_loader()
            return next(self.dataloader)[0]
