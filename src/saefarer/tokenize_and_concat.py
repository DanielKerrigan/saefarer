from typing import Iterator, Optional, Tuple, Union

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


def _prepend_token_if_not_there(token_id: int, tokens: torch.Tensor):
    """Add token_id to the start of tokens unless it is already there."""
    if tokens[0] == token_id:
        return tokens

    return torch.cat(
        (
            torch.tensor([token_id], dtype=torch.long, device=tokens.device),
            tokens,
        ),
        dim=0,
    )


def _add_tokens_to_batch(
    batch: Union[torch.Tensor, None],
    tokens: torch.Tensor,
    context_size: int,
    is_start_of_sequence: bool,
    begin_batch_token_id: Optional[int],
    begin_sequence_token_id: Optional[int],
    sequence_separator_token_id: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add as many tokens to the batch as possible.
    Return the batch and any tokens which were not added."""
    original_tokens = tokens

    # prepend start of sequence token if needed
    if is_start_of_sequence and begin_sequence_token_id is not None:
        tokens = _prepend_token_if_not_there(begin_sequence_token_id, tokens)

    # start of new batch
    if batch is None:
        # prepend start of batch token if needed
        if begin_batch_token_id is not None:
            tokens = _prepend_token_if_not_there(begin_batch_token_id, tokens)

        return tokens[:context_size], tokens[context_size:]

    # concatenating batches
    if sequence_separator_token_id is not None:
        tokens = _prepend_token_if_not_there(sequence_separator_token_id, tokens)

    tokens_needed = context_size - batch.shape[0]
    batch = torch.cat([batch, tokens[:tokens_needed]])

    remaining_tokens = tokens[tokens_needed:]

    # it's possible we've prepended two tokens, but only removed one.
    # if so, we should only return the original tokens.

    if len(remaining_tokens) > len(original_tokens):
        remaining_tokens = original_tokens

    return batch, remaining_tokens


@torch.no_grad()
def tokenize_and_concat_iterator(
    dataset_iterator: Iterator[str],
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    context_size: int,
    begin_batch_token_id: Optional[int],
    begin_sequence_token_id: Optional[int],
    sequence_separator_token_id: Optional[int],
) -> Iterator[torch.Tensor]:
    """Iterator over context_size length sequences of tokens."""
    batch: Union[torch.Tensor, None] = None

    for row in dataset_iterator:
        tokens = tokenizer(row, return_tensors="pt").input_ids.squeeze(0)

        remaining_tokens = tokens

        is_start_of_sequence = True

        while len(remaining_tokens) > 0:
            batch, remaining_tokens = _add_tokens_to_batch(
                batch=batch,
                tokens=remaining_tokens,
                context_size=context_size,
                is_start_of_sequence=is_start_of_sequence,
                begin_batch_token_id=begin_batch_token_id,
                begin_sequence_token_id=begin_sequence_token_id,
                sequence_separator_token_id=sequence_separator_token_id,
            )

            is_start_of_sequence = False

            if batch.shape[0] == context_size:
                yield batch
                batch = None


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    column_name: str,
    context_size: int,
    begin_batch_token_id: Optional[int] = None,
    begin_sequence_token_id: Optional[int] = None,
    sequence_separator_token_id: Optional[int] = None,
    num_proc: Optional[int] = None,
) -> Dataset:
    def tokenization(examples):
        tokens = list(
            tokenize_and_concat_iterator(
                dataset_iterator=iter(examples[column_name]),
                tokenizer=tokenizer,
                context_size=context_size,
                begin_batch_token_id=begin_batch_token_id,
                begin_sequence_token_id=begin_sequence_token_id,
                sequence_separator_token_id=sequence_separator_token_id,
            )
        )
        return {"input_ids": tokens}

    tokenized_dataset = dataset.map(
        tokenization,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
    )
    tokenized_dataset.set_format(type="torch", columns=["input_ids"])

    return tokenized_dataset
