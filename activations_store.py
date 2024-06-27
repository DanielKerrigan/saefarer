"""ActivationsStore"""

import torch
from datasets import Dataset
from einops import rearrange
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from config import Config


class ActivationsStore:
    """ActivationsStore"""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        dataset: Dataset,
        cfg: Config,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.cfg = cfg

        self.iterable_dataset = iter(dataset)

        self.store_index = 0

        self.store = torch.zeros(
            (
                cfg.n_batches_in_store
                * cfg.lm_batch_size_sequences
                * cfg.lm_sequence_length,
                cfg.d_in,
            ),
            dtype=torch.bfloat16,
            requires_grad=False,
        )

        self.refresh(self.cfg.n_batches_in_store)

    @torch.no_grad()
    def next(self):
        """Get the next batch of activations for SAE training."""
        activations = self.store[
            self.store_index : self.store_index + self.cfg.sae_batch_size_tokens
        ]

        self.store_index += self.cfg.sae_batch_size_tokens

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

        n_tokens_in_lm_batch = (
            self.cfg.lm_batch_size_sequences * self.cfg.lm_sequence_length
        )

        for i in range(n_batches):
            tokens = self.get_batch_tokens()
            activations = self.get_activations(tokens)

            start = i * n_tokens_in_lm_batch
            end = start + n_tokens_in_lm_batch
            self.store[start:end] = activations

        self.store = self.store[torch.randperm(self.store.shape[0])]

    def get_batch_tokens(self):
        """Get batch of tokens from the dataset."""

        batch_sequences = []

        cur_sequence = []
        cur_sequence_length = 0

        while len(batch_sequences) < self.cfg.lm_batch_size_sequences:
            tokens = self.get_next_tokens()

            while (
                tokens.shape[0] > 0
                and len(batch_sequences) < self.cfg.lm_batch_size_sequences
            ):
                space_left_in_seq = self.cfg.lm_sequence_length - cur_sequence_length

                n_tokens_to_add = min(tokens.shape[0], space_left_in_seq)

                cur_sequence.append(tokens[:n_tokens_to_add])
                cur_sequence_length += n_tokens_to_add
                tokens = tokens[n_tokens_to_add:]

                if cur_sequence_length == self.cfg.lm_sequence_length:
                    sequence = torch.cat(cur_sequence, dim=0)
                    batch_sequences.append(sequence)
                    cur_sequence = []
                    cur_sequence_length = 0

                    if (
                        tokens.shape[0] > 0
                        and self.cfg.prepend_bos_token
                        and tokens[0] != self.tokenizer.bos_token_id
                    ):
                        tokens = torch.cat(
                            (torch.tensor([self.tokenizer.bos_token_id]), tokens), dim=0
                        )

        batch = torch.stack(batch_sequences, dim=0)

        return batch

    def get_next_tokens(self):
        """Get tokens for the next item in the dataset."""
        text = next(self.iterable_dataset)["text"]

        if self.cfg.prepend_bos_token:
            text = self.tokenizer.bos_token + text

        tokens = self.tokenizer(text, return_tensors="pt")["input_ids"].squeeze()
        return tokens

    @torch.no_grad()
    def get_activations(self, batch_tokens):
        """Get activations for tokens."""
        batch_output = self.model(batch_tokens, output_hidden_states=True)
        batch_activations = batch_output.hidden_states[self.cfg.hidden_state_index]
        flat_activations = rearrange(
            batch_activations, "batches seq_len d_in -> (batches seq_len) d_in"
        )
        return flat_activations
