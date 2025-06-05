from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import (
        BertTokenizer,
        BertTokenizerFast,
        RobertaTokenizer,
        RobertaTokenizerFast,
    )


class HuggingFaceBertTokenizerAdapter:
    def __init__(self, tokenizer: "BertTokenizer | BertTokenizerFast"):
        self.tokenizer = tokenizer
        self.all_special_ids = tokenizer.all_special_ids

    def decode(self, token_ids: list[int]) -> str:
        tokens = []

        for token in self.tokenizer.convert_ids_to_tokens(token_ids):
            if token.startswith("##"):
                tokens.append(token[2:])
            else:
                tokens.append(" " + token)

        return "".join(tokens)


class HuggingFaceRobertaTokenizerAdapter:
    def __init__(self, tokenizer: "RobertaTokenizer | RobertaTokenizerFast"):
        self.tokenizer = tokenizer
        self.all_special_ids = tokenizer.all_special_ids

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)
