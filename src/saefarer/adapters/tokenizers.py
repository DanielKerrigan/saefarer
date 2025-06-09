from typing import TYPE_CHECKING

from saefarer.protocols import EncodingOutput

if TYPE_CHECKING:
    from transformers import (
        BertTokenizer,
        BertTokenizerFast,
        RobertaTokenizer,
        RobertaTokenizerFast,
    )


class HuggingFaceBertTokenizerAdapter:
    def __init__(
        self,
        tokenizer: "BertTokenizer | BertTokenizerFast",
    ):
        self.tokenizer = tokenizer
        self.pad_token_id: int = tokenizer.pad_token_id  # type: ignore

    def encode(
        self,
        text: str,
        max_length: int,
    ):
        output = self.tokenizer(
            text,
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )

        token_ids: list[int] = output["input_ids"]  # type: ignore
        attention_mask: list[int] = output["attention_mask"]  # type: ignore

        return EncodingOutput(token_ids=token_ids, attention_mask=attention_mask)

    def decode(self, token_ids: list[int]) -> str:
        tokens = []

        for token in self.tokenizer.convert_ids_to_tokens(token_ids):
            if token.startswith("##"):
                tokens.append(token[2:])
            else:
                tokens.append(" " + token)

        return "".join(tokens)


class HuggingFaceRobertaTokenizerAdapter:
    def __init__(
        self,
        tokenizer: "RobertaTokenizer | RobertaTokenizerFast",
    ):
        self.tokenizer = tokenizer
        self.pad_token_id: int = tokenizer.pad_token_id  # type: ignore

    def encode(
        self,
        text: str,
        max_length: int,
    ):
        output = self.tokenizer(
            text,
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )

        token_ids: list[int] = output["input_ids"]  # type: ignore
        attention_mask: list[int] = output["attention_mask"]  # type: ignore

        return EncodingOutput(token_ids=token_ids, attention_mask=attention_mask)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)
