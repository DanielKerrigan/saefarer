from typing import TYPE_CHECKING

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
        self.pad_token_id = tokenizer.pad_token_id

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

        return {
            "token_ids": output["input_ids"],
            "attention_mask": output["attention_mask"],
        }

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
        self.pad_token_id = tokenizer.pad_token_id

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

        return {
            "token_ids": output["input_ids"],
            "attention_mask": output["attention_mask"],
        }

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)
