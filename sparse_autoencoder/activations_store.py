""""""

from typing import Union

import torch
from datasets import Dataset, IterableDataset
from einops import rearrange
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from sparse_autoencoder.config import Config
from sparse_autoencoder.constants import DTYPES


class ActivationsStore:
    """
    This class is used to provide model activations from a given
    layer to train the SAE on.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        dataset: Union[Dataset, IterableDataset],
        cfg: Config,
    ):
        self.dtype = DTYPES[cfg.dtype]
        self.device = torch.device(cfg.device)

        self.model = model

        self.tokenizer = tokenizer
        self.dataset = dataset
        self.cfg = cfg

        self.iterable_dataset = iter(dataset)

        self.store = torch.zeros(
            (
                cfg.n_batches_in_store
                * cfg.model_batch_size_sequences
                * cfg.model_sequence_length,
                cfg.d_in,
            ),
            dtype=self.dtype,
            requires_grad=False,
            device=self.device,
        )

        # our current position in the store
        self.store_index = 0

        self.store_filled_once = False

    @torch.no_grad()
    def next(self) -> torch.Tensor:
        """
        Get the next batch of model activations for SAE training.
        This batch has dimensions [sae_batch_size_tokens, d_in].
        """

        # if this is the first time next is being called, fill the store
        if not self.store_filled_once:
            self.refresh(self.cfg.n_batches_in_store)
            self.store_filled_once = True

        # get a batch of activations from the store
        activations = self.store[
            self.store_index : self.store_index + self.cfg.sae_batch_size_tokens
        ]

        # keep track of how much of the store we have read
        self.store_index += self.cfg.sae_batch_size_tokens

        # if we have read more than half of the store, then re-fill the store
        if (
            self.store_index
            > (self.store.shape[0] // 2) - self.cfg.sae_batch_size_tokens
        ):
            self.refresh(self.cfg.n_batches_in_store // 2)

        return activations

    @torch.no_grad()
    def refresh(self, n_batches):
        """Refresh the store."""
        self.store_index = 0

        n_tokens_in_model_batch = (
            self.cfg.model_batch_size_sequences * self.cfg.model_sequence_length
        )

        # overwrite the first n_batches of the store,
        # which have already been used
        for i in range(n_batches):
            tokens = self.get_batch_tokens()
            activations = self.get_activations(tokens)

            start = i * n_tokens_in_model_batch
            end = start + n_tokens_in_model_batch
            self.store[start:end] = activations

        # shuffle the store
        self.store = self.store[torch.randperm(self.store.shape[0])]

    def get_batch_tokens(self) -> torch.Tensor:
        """Get batch of tokens from the dataset."""

        batch_sequences = []

        cur_sequence = []
        cur_sequence_length = 0

        while len(batch_sequences) < self.cfg.model_batch_size_sequences:
            # get tokens for next item in dataset
            tokens = self.get_next_tokens()

            # add as much of these tokens to the current sequence as we can
            while (
                tokens.shape[0] > 0
                and len(batch_sequences) < self.cfg.model_batch_size_sequences
            ):
                space_left_in_seq = self.cfg.model_sequence_length - cur_sequence_length

                n_tokens_to_add = min(tokens.shape[0], space_left_in_seq)

                cur_sequence.append(tokens[:n_tokens_to_add])
                cur_sequence_length += n_tokens_to_add
                tokens = tokens[n_tokens_to_add:]

                # if the current sequence is full, add it to the batch
                if cur_sequence_length == self.cfg.model_sequence_length:
                    sequence = torch.cat(cur_sequence, dim=0)
                    batch_sequences.append(sequence)
                    cur_sequence = []
                    cur_sequence_length = 0

                    # if there are still tokens left, prepend the bos token
                    if (
                        tokens.shape[0] > 0
                        and self.cfg.prepend_bos_token
                        and tokens[0] != self.tokenizer.bos_token_id
                    ):
                        tokens = torch.cat(
                            (
                                torch.tensor(
                                    [self.tokenizer.bos_token_id], device=self.device
                                ),
                                tokens,
                            ),
                            dim=0,
                        )

        batch = torch.stack(batch_sequences, dim=0)

        return batch

    def get_next_tokens(self) -> torch.Tensor:
        """Get tokens for the next item in the dataset."""
        text = next(self.iterable_dataset)["text"]

        if self.cfg.prepend_bos_token:
            text = self.tokenizer.bos_token + text

        tokens = self.tokenizer(text, return_tensors="pt").input_ids.squeeze(0)

        return tokens.to(self.device)

    @torch.no_grad()
    def get_activations(self, batch_tokens) -> torch.Tensor:
        """Get activations for tokens."""
        batch_output = self.model(batch_tokens, output_hidden_states=True)
        batch_activations = batch_output.hidden_states[self.cfg.hidden_state_index]
        flat_activations = rearrange(
            batch_activations, "batches seq_len d_in -> (batches seq_len) d_in"
        )
        return flat_activations
